import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguedashplayerstats
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import plotly.graph_objects as go

# --- Model Definition ---
class PlayerAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    # Official NBA stats
    df_stats = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25").get_data_frames()[0]
    cols = ['PLAYER_NAME', 'AGE', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS','TEAM_ABBREVIATION','TEAM_ID']
    df_stats = df_stats[cols]

    # Convert cumulative stats to per-game averages
    stats_to_average = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    for stat in stats_to_average:
        df_stats[stat] = df_stats[stat] / df_stats['GP']

    def keep_tot_rows(df):
        if 'Tm' in df.columns:
            return df.sort_values(by='Tm', ascending=False).drop_duplicates(subset='Player', keep='first')
        return df.drop_duplicates(subset='Player')

    # Load Advanced Metrics
    url_advanced = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html"
    df_adv = pd.read_html(url_advanced)[0]
    df_adv = df_adv[df_adv['Player'] != 'Player']
    df_adv = keep_tot_rows(df_adv)
    df_adv = df_adv[['Player', 'TS%', 'USG%', 'PER', 'BPM', 'VORP']]

    # Load Per-Minute Stats
    url_per_min = "https://www.basketball-reference.com/leagues/NBA_2025_per_minute.html"
    df_per_min = pd.read_html(url_per_min)[0]
    df_per_min = df_per_min[df_per_min['Player'] != 'Player']
    df_per_min = keep_tot_rows(df_per_min)
    df_per_min = df_per_min[['Player', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV']]
    df_per_min.rename(columns={
        'PTS': 'PTS_per36', 'AST': 'AST_per36', 'TRB': 'REB_per36',
        'STL': 'STL_per36', 'BLK': 'BLK_per36', 'TOV': 'TOV_per36'
    }, inplace=True)

    # Merge
    df_extra = df_adv.merge(df_per_min, on='Player', how='inner')

    # Convert all columns except 'Player' to numeric
    for col in df_extra.columns:
        if col != 'Player':
            df_extra[col] = pd.to_numeric(df_extra[col], errors='coerce')

    # Drop rows with any NaN (caused by bad values like '—')
    df_extra = df_extra.dropna()

    # Advanced metrics (manually simplified sample)
    df_extra['Player'] = df_extra['Player'].str.lower()
    df_stats['Player'] = df_stats['PLAYER_NAME'].str.lower()

    df_merged = df_stats.merge(df_extra, on='Player', how='inner')
    names = df_merged['PLAYER_NAME'].tolist()
    X = df_merged.select_dtypes(include='number')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_merged, names, scaler, X_scaled

df_merged, names, scaler, X_scaled = load_data()

# --- Train or Load Model ---
@st.cache_resource
def train_model(X_scaled):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model = PlayerAutoencoder(input_dim=X_tensor.shape[1], embedding_dim=12)
    optimizer = Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=True)
    for epoch in range(50):
        for (batch,) in loader:
            recon, _ = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        _, embeddings = model(X_tensor)

    return model, embeddings

model, embeddings = train_model(X_scaled)

# --- Similarity Search ---
def find_similar_players(player_name, names, embeddings, top_k=5):
    if player_name not in names:
        return []

    idx = names.index(player_name)
    target_embedding = embeddings[idx]

    similarities = torch.nn.functional.cosine_similarity(
        target_embedding.unsqueeze(0), embeddings
    )
    sim_scores = similarities.tolist()

    top_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)
    top_indices = [i for i in top_indices if i != idx][:top_k]

    results = [(names[i], sim_scores[i]) for i in top_indices]
    return results

# --- Fuzzy Search Helper ---
def suggest_player(name, names):
    from difflib import get_close_matches
    matches = get_close_matches(name, names, n=1, cutoff=0.5)
    return matches

# --- Image Fetch ---
def get_player_image_url(name):
    result = players.find_players_by_full_name(name)
    if result:
        player_id = result[0]["id"]
        return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    return None

# --- Team Name Mapping ---
TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls",
    "CHA": "Charlotte Hornets",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

# --- Streamlit UI ---
st.title("🏀 NBA Player Similarity Search")
player_query = st.text_input("Enter a player's name:", placeholder="LeBron James")

if player_query:
    suggested = suggest_player(player_query, names)
    if suggested:
        selected = suggested[0]
        # st.success(f"Showing results for **{selected.title()}**")
        results = find_similar_players(selected, names, embeddings)
        if results:
            # Player image + stats display
            img_url = get_player_image_url(selected)
            player_row = df_merged[df_merged['PLAYER_NAME'] == selected]

            team_abbr = player_row['TEAM_ABBREVIATION'].iloc[0]
            team_full = TEAM_NAME_MAP.get(team_abbr, team_abbr)

            st.subheader(f"{selected.title()} of {team_full}")

            col1, col2 = st.columns([1, 2])

            with col1:
                if img_url:
                    # st.markdown(
                    #     f"""
                    #     <div style="height: 225px; display: flex; align-items: center;">
                    #         <img src="{img_url}" style="max-height: 100%; width: auto;">
                    #     </div>
                    #     """,
                    #     unsafe_allow_html=True
                    # )
                    st.image(img_url, use_container_width=True)


                else:
                    st.write("Image not found.")

            with col2:
                main_stats = player_row[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']].iloc[0]

                row1 = st.columns(3)
                row2 = st.columns(3)

                stats_list = list(main_stats.items())
                for i in range(3):
                    stat, value = stats_list[i]
                    row1[i].metric(label=stat, value=f"{value:.1f}", border=False)
                for i in range(3, 6):
                    stat, value = stats_list[i]
                    row2[i - 3].metric(label=stat, value=f"{value:.1f}", border=False)

            # Define which stats you've already shown
            shown_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']

            # Get full numeric stats for the player
            full_stats = player_row.select_dtypes(include='number').copy()

            # Remaining Stats
            remaining_stats = full_stats.drop(columns=shown_stats, errors='ignore').reset_index(drop=True)

            # Converting total minutes to minutes per game
            if 'MIN' in remaining_stats.columns and 'GP' in remaining_stats.columns:
                remaining_stats['MIN_per_game'] = remaining_stats['MIN'] / remaining_stats['GP']
                remaining_stats.drop(columns=['MIN'], inplace=True)

            # Drop identifying fields too (like GP or AGE) if you don’t want them
            remaining_stats.drop(columns=['AGE', 'GP','TEAM_ID'], inplace=True, errors='ignore')

            # Creating PER36 dataframe
            per36_cols = [col for col in remaining_stats.columns if col.endswith('_per36')]
            per36_df = remaining_stats[per36_cols].copy()

            with st.expander("📊 More Stats"):
                # Show the rest of the stats
                st.write("##### Additional Stats")
                st.dataframe(remaining_stats, use_container_width=True, hide_index=True)

                # Show PER36 dataframe
                st.write("##### PER36 Stats")
                st.dataframe(per36_df, use_container_width=True, hide_index=True)

            # st.divider()

            st.subheader("Top 5 Similar Players")
            for name, score in results:
                row = df_merged[df_merged['PLAYER_NAME'] == name].copy()
                row_stats = row[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']].iloc[0]
                sim_score = f"{score:.3f}"

                team_abbr = row['TEAM_ABBREVIATION'].iloc[0]
                team_full = TEAM_NAME_MAP.get(team_abbr, team_abbr)

                with st.expander(f"🎯 {name} of {team_full}", expanded=True):
                    similarity_percentage = float(sim_score)*100
                    similarity_display =  f"{similarity_percentage:.1f}%"
                    # st.subheader(f"Similarity: {similarity_display}",divider=True)

                    img = get_player_image_url(name)
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.badge(f"Similarity: {similarity_display}",color="blue",width="stretch")

                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.write("Image not found")

                    with col2:
                        row1 = st.columns(3)
                        row2 = st.columns(3)

                        stats_list = list(row_stats.items())
                        for i in range(3):
                            stat, value = stats_list[i]
                            base_val = player_row[stat].values[0]  # from searched player
                            delta = round(value - base_val, 1)
                            delta_color = "inverse" if stat == "TOV" else "normal"
                            row1[i].metric(label=stat, value=f"{value:.1f}", delta=f"{delta:+.1f}",delta_color=delta_color)

                        for i in range(3, 6):
                            stat, value = stats_list[i]
                            base_val = player_row[stat].values[0]
                            delta = round(value - base_val, 1)
                            delta_color = "inverse" if stat == "TOV" else "normal"
                            row2[i - 3].metric(label=stat, value=f"{value:.1f}", delta=f"{delta:+.1f}",delta_color=delta_color)

                    # Get full numeric stats for the player
                    full_stats = row.select_dtypes(include='number').copy()

                    # Remaining Stats
                    remaining_stats = full_stats.drop(columns=shown_stats, errors='ignore').reset_index(drop=True)

                    # Converting total minutes to minutes per game
                    if 'MIN' in remaining_stats.columns and 'GP' in remaining_stats.columns:
                        remaining_stats['MIN_per_game'] = remaining_stats['MIN'] / remaining_stats['GP']
                        remaining_stats.drop(columns=['MIN'], inplace=True)

                    # Drop identifying fields too (like GP or AGE) if you don’t want them
                    remaining_stats.drop(columns=['AGE', 'GP','TEAM_ID'], inplace=True, errors='ignore')

                    # Creating PER36 dataframe
                    per36_cols = [col for col in remaining_stats.columns if col.endswith('_per36')]
                    per36_df = remaining_stats[per36_cols].copy()

                    with st.expander("See More"):
                        tab1, tab2 = st.tabs(["📊 Stats", "📈 Charts"])

                        with tab1:
                            # Show the rest of the stats
                            st.write("##### Additional Stats")
                            st.dataframe(remaining_stats, use_container_width=True, hide_index=True)

                            # Show PER36 dataframe
                            st.write("##### PER36 Stats")
                            st.dataframe(per36_df, use_container_width=True, hide_index=True)
                        
                        with tab2:
                            # Radar Chart
                            # Define radar categories
                            categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']

                            # Get stat values
                            searched_values = [float(player_row[stat]) for stat in categories]
                            similar_values = [float(row[stat]) for stat in categories]

                            # Create radar chart
                            fig = go.Figure()

                            fig.add_trace(go.Scatterpolar(
                                r=searched_values,
                                theta=categories,
                                fill='toself',
                                name=selected,
                                line_color='blue'
                            ))
                            fig.add_trace(go.Scatterpolar(
                                r=similar_values,
                                theta=categories,
                                fill='toself',
                                name=name,
                                line_color='orange'
                            ))

                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, max(searched_values + similar_values) * 1.1])
                                ),
                                showlegend=True,
                                margin=dict(l=20, r=20, t=30, b=20),
                                height=400,
                                title='Radar Chart',
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.divider()

                            # Bar Chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=categories,
                                x=[player_row[stat].values[0] for stat in categories],
                                name=selected,
                                orientation='h',
                                marker_color='blue'
                            ))
                            fig.add_trace(go.Bar(
                                y=categories,
                                x=[row[stat].values[0] for stat in categories],  # <- corrected here
                                name=name,
                                orientation='h',
                                marker_color='orange'
                            ))

                            fig.update_layout(
                                barmode='group',
                                title='Bar Chart',
                                height=400,
                                margin=dict(l=30, r=30, t=40, b=30)
                            )

                            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No close player match found. Try a different name.")
    else:
        st.error("No close player match found. Try a different name.")
