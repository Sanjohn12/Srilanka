import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🌿 Sri Lanka Environmental Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN STYLING ---
st.markdown(
    """
    <style>
    .block-container {
        background-color: white;
        color: black;
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

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('Srilanka/District_Dashboard/District_Data_Modified.csv')
    gdf = gpd.read_file("Srilanka/District_Dashboard/lka_admbnda_adm2_slsd_20220816.shp")
    gdf = gdf.to_crs(epsg=4326)  # make sure it's in WGS84
    return df, gdf

df, gdf = load_data()

# --- NORMALIZATION ---
features = df.columns[1:]
scaler = MinMaxScaler()
scaled_data = df.copy()
scaled_data[features] = scaler.fit_transform(df[features])

good = [
    'Rain_dist_Mean_Rainfall_mm',
    'District_Mean_NDVI_2020_2025_Mean_NDVI',
    'canopy_dist_Mean_Canopy_Height',
    'treeloss_treecover_Mean_TreeCover2000'
]
bad = [
    'co_dist_Mean_CO',
    'District_Mean_NO2_2019_2024_Mean_NO2',
    'treeloss_treecover_Forest_Loss_km2',
    'District_Mean_SI_2020_2025_Mean_SI'
]
for b in bad:
    scaled_data[b] = 1 - scaled_data[b]

scaled_data['Environmental_Score'] = scaled_data[good + bad].mean(axis=1)
scaled_data['Rank'] = scaled_data['Environmental_Score'].rank(ascending=False, method='min').astype(int)

# --- MERGE GEO & DATA ---
merged = gdf.merge(df, on='ADM2_EN')
merged = merged.merge(scaled_data[['ADM2_EN', 'Environmental_Score', 'Rank']], on='ADM2_EN')

# --- GEOJSON CONVERSION ---
merged['id'] = merged.index.astype(str)
geojson_data = json.loads(merged.to_json())

# --- SIDEBAR ---
st.sidebar.header("🌿 Filter and Explore")

district_options = st.sidebar.multiselect(
    "Select districts for radar chart:",
    options=scaled_data['ADM2_EN'],
    default=scaled_data['ADM2_EN'][:3].tolist()
)

selected_param = st.sidebar.selectbox(
    "Select parameter to map:",
    options=features,
    index=0
)

# --- TITLE ---
st.title("🌿 Sri Lanka District Environmental Ranking Dashboard")

# --- RANKING TABLE ---
st.subheader("📊 District Rankings")
ranked = scaled_data[['Rank', 'ADM2_EN', 'Environmental_Score']].sort_values('Rank').reset_index(drop=True)
ranked['Environmental_Score'] = ranked['Environmental_Score'].round(3)
ranked['Rank Badge'] = ranked['Rank'].apply(lambda r: {1: "🥇", 2: "🥈", 3: "🥉"}.get(r, ""))

st.dataframe(
    ranked.style.highlight_max(subset=['Environmental_Score'], color='#85C1E9')
          .format({"Environmental_Score": "{:.3f}"}).set_properties(**{'text-align': 'center'}),
    use_container_width=True
)

st.markdown("🏆 **Top 3 Districts:**")
for _, row in ranked.head(3).iterrows():
    st.markdown(f"{row['Rank Badge']} **{row['ADM2_EN']}** — Score: {row['Environmental_Score']}")

# --- BAR CHART ---
st.subheader("🏆 Scores by District")
fig_bar = px.bar(
    ranked, x='Environmental_Score', y='ADM2_EN',
    orientation='h', color='Environmental_Score',
    color_continuous_scale='YlGnBu'
)
fig_bar.update_layout(yaxis=dict(autorange='reversed'))
st.plotly_chart(fig_bar, use_container_width=True)

# --- RADAR CHART ---
if district_options:
    st.subheader("🕸️ District Radar Comparison")
    fig_radar = go.Figure()
    for dist in district_options:
        row = scaled_data[scaled_data['ADM2_EN'] == dist].iloc[0][features]
        fig_radar.add_trace(go.Scatterpolar(
            r=row.values, theta=row.index, fill='toself', name=dist
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Environmental Parameters"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- MAP + PARAM TABLE ---
st.subheader(f"🔍 Values & Map for: {selected_param}")
param_table = merged[['ADM2_EN', selected_param]].sort_values(selected_param, ascending=False)
st.dataframe(
    param_table.style.background_gradient(subset=[selected_param], cmap='YlGnBu')
                .set_properties(**{'text-align': 'center'}),
    use_container_width=True
)

fig_map = px.choropleth_mapbox(
    merged,
    geojson=geojson_data,
    locations='id',
    color=selected_param,
    hover_name='ADM2_EN',
    color_continuous_scale='YlGnBu',
    mapbox_style="carto-positron",
    zoom=6.5,
    center={"lat": 7.8731, "lon": 80.7718},
    opacity=0.8,
    labels={selected_param: selected_param},
    height=680
)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)
