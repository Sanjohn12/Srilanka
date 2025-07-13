import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🌿 Sri Lanka Environmental Ranking Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD AND CACHE RAW DATA ---
@st.cache_data(show_spinner=True)
def load_raw_data():
    df = pd.read_csv("District_Dashboard/District_Data_Modified.csv")
    shp_path = "District_Dashboard/lka_admbnda_adm2_slsd_20220816.shp"
    gdf = gpd.read_file(shp_path)
    return df, gdf

# --- PREPROCESSING & MERGING (CACHE THIS TOO!) ---
@st.cache_data(show_spinner=True)
def preprocess_and_merge(df, gdf):
    features = df.columns[1:]
    scaler = MinMaxScaler()
    scaled_data = df.copy()
    scaled_data[features] = scaler.fit_transform(df[features])

    good = ['Rain_dist_Mean_Rainfall_mm', 'District_Mean_NDVI_2020_2025_Mean_NDVI',
            'canopy_dist_Mean_Canopy_Height', 'treeloss_treecover_Mean_TreeCover2000']
    bad = ['co_dist_Mean_CO', 'District_Mean_NO2_2019_2024_Mean_NO2',
           'treeloss_treecover_Forest_Loss_km2', 'District_Mean_SI_2020_2025_Mean_SI']

    for b in bad:
        scaled_data[b] = 1 - scaled_data[b]

    scaled_data['Environmental_Score'] = scaled_data[good + bad].mean(axis=1)
    scaled_data['Rank'] = scaled_data['Environmental_Score'].rank(ascending=False, method='min').astype(int)

    merged = gdf.merge(df, on='ADM2_EN')
    merged = merged.merge(scaled_data[['ADM2_EN', 'Environmental_Score', 'Rank']], on='ADM2_EN')

    return features, scaled_data, merged

# Load data once and preprocess once (cached)
df, gdf = load_raw_data()
features, scaled_data, merged = preprocess_and_merge(df, gdf)

# --- SIDEBAR ---
st.sidebar.header("🌿 Filter and Explore")

district_options = st.sidebar.multiselect(
    "Select districts for profile comparison (Radar Chart):",
    options=scaled_data['ADM2_EN'],
    default=scaled_data['ADM2_EN'][:3].tolist()
)

selected_param = st.sidebar.selectbox(
    "Select parameter to view district values and map:",
    options=features,
    index=0
)

# --- MAIN PAGE ---

# Glassmorphism style for main container (your original style)
st.markdown(
    """
    <style>
    .main {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: #111;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .dataframe th {
        background-color: #4CAF50 !important;
        color: white !important;
        text-align: center !important;
    }
    .dataframe td {
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌿 Sri Lanka District Environmental Ranking Dashboard")

# Ranking table
st.subheader("📊 District Rankings by Environmental Score")
ranked = scaled_data[['Rank', 'ADM2_EN', 'Environmental_Score']].sort_values('Rank').reset_index(drop=True)
ranked['Environmental_Score'] = ranked['Environmental_Score'].round(3)

def rank_badge(rank):
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    else:
        return ""

ranked['Rank Badge'] = ranked['Rank'].apply(rank_badge)

st.dataframe(
    ranked.style.highlight_max(subset=['Environmental_Score'], color='#85C1E9')
          .format({"Environmental_Score": "{:.3f}"})
          .set_properties(**{'text-align': 'center'}),
    use_container_width=True
)

st.markdown("🏆 **Top 3 Districts:**")
for idx, row in ranked.head(3).iterrows():
    st.markdown(f"{row['Rank Badge']} **{row['ADM2_EN']}** — Score: {row['Environmental_Score']}")

# Bar chart for ranking
st.subheader("🏆 Environmental Scores by District")
fig_bar = px.bar(
    ranked,
    x='Environmental_Score',
    y='ADM2_EN',
    orientation='h',
    color='Environmental_Score',
    color_continuous_scale='YlGnBu',
    labels={'Environmental_Score': 'Env. Score', 'ADM2_EN': 'District'},
    title="District Environmental Score Ranking"
)
fig_bar.update_layout(yaxis=dict(autorange='reversed'))
st.plotly_chart(fig_bar, use_container_width=True)

# Radar chart for comparison
if district_options:
    st.subheader("🕸️ District Profile Comparison (Radar Chart)")
    fig_radar = go.Figure()
    for dist in district_options:
        row = scaled_data[scaled_data['ADM2_EN'] == dist].iloc[0][features]
        fig_radar.add_trace(go.Scatterpolar(
            r=row.values,
            theta=row.index,
            fill='toself',
            name=dist
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Environmental Parameters"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# District-wise parameter table & map (show original values, NOT scaled)
st.subheader(f"🔍 District-wise Values & Map for Parameter: {selected_param}")

param_table = merged[['ADM2_EN', selected_param]].sort_values(selected_param, ascending=False).reset_index(drop=True)

st.dataframe(
    param_table.style.background_gradient(
        subset=[selected_param],
        cmap='YlGnBu'
    ).set_properties(**{'text-align': 'center'}),
    use_container_width=True
)

fig_map = px.choropleth_mapbox(
    merged,
    geojson=merged.geometry,
    locations=merged.index,
    color=selected_param,
    hover_name='ADM2_EN',
    color_continuous_scale='YlGnBu',
    mapbox_style="carto-positron",
    zoom=7,
    center={"lat": 7.8731, "lon": 80.7718},
    opacity=0.7,
    labels={selected_param: selected_param}
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig_map, use_container_width=True)
