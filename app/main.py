#====================================================
# Application Streamlit pour l'analyse du Superstore
# Kodjo Jean DEGBEVI
#====================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

CURRENT_FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_FILE_DIR.parent
data_path = ROOT_DIR / "data" / "processed" / "superstore_processed.csv"
sys.path.append(str(ROOT_DIR))

from src.utils import get_us_state_abbrev

st.set_page_config(page_title="Superstore Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Year'] = df['Order Date'].dt.year
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Filtres")

selected_years = st.sidebar.multiselect(
    "Années", 
    options=sorted(df['Order Year'].unique()), 
    default=sorted(df['Order Year'].unique())
)

selected_regions = st.sidebar.multiselect(
    "Régions", 
    options=sorted(df['Region'].unique()), 
    default=sorted(df['Region'].unique())
)

selected_categories = st.sidebar.multiselect(
    "Catégories", 
    options=sorted(df['Category'].unique()), 
    default=sorted(df['Category'].unique())
)

if not selected_years: selected_years = df['Order Year'].unique()
if not selected_regions: selected_regions = df['Region'].unique()
if not selected_categories: selected_categories = df['Category'].unique()

filtered_df = df[
    (df['Order Year'].isin(selected_years)) &
    (df['Region'].isin(selected_regions)) &
    (df['Category'].isin(selected_categories))
]

# --- HEADER ET KPIS ---
st.title("Dashboard - Superstore")

col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
marge_globale = total_profit / total_sales if total_sales > 0 else 0
total_customers = filtered_df['Customer ID'].nunique()

# Amplitude temporelle
if not filtered_df.empty:
    date_min = filtered_df['Order Date'].min()
    date_max = filtered_df['Order Date'].max()
    days = max(1, (date_max - date_min).days)
    nb_annees = max(1, days / 365.25)
else:
    nb_annees = 1

avg_sales = total_sales / nb_annees
avg_profit = total_profit / nb_annees
avg_customers = total_customers / nb_annees

col1.metric("Chiffre d'Affaires", f"${total_sales:,.0f}", f"{avg_sales:,.0f} $ / an", delta_color="off")
col2.metric("Bénéfice Net", f"${total_profit:,.0f}", f"{avg_profit:,.0f} $ / an", delta_color="off")
col3.metric("Marge Globale", f"{marge_globale:.2%}")
col4.metric("Clients Uniques", f"{total_customers:,}", f"~ {int(avg_customers):,} / an", delta_color="off")

st.markdown("---")

# --- ONGLETS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Performance Géographique", 
    "2. Rentabilité par Produit", 
    "3. Sensibilité aux Remises", 
    "4. Valeur Client"
])

with tab1:
    st.subheader("Cartographie de la Rentabilité")
    
    us_state_abbrev = get_us_state_abbrev()

    df_state = filtered_df.groupby('State', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'})
    df_state['State Code'] = df_state['State'].map(us_state_abbrev)

    fig_map = px.choropleth(
        df_state, locations='State Code', locationmode="USA-states", color='Profit',
        scope="usa", hover_name='State',
        hover_data={'State Code': False, 'Sales': ':$,.0f', 'Profit': ':$,.0f', 'Discount': ':.1%'},
        color_continuous_scale='RdBu', color_continuous_midpoint=0
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_map, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("États Extrêmes (Top vs Flop)")

    top_10 = df_state.sort_values(by='Profit', ascending=False).head(5)
    flop_10 = df_state.sort_values(by='Profit', ascending=True).head(5)

    fig_tf = make_subplots(rows=1, cols=2, subplot_titles=("Top 5 - Profits Stratégiques", "Flop 5 - Destructeurs de Valeur"))
    fig_tf.add_trace(go.Bar(x=top_10['State'], y=top_10['Profit'], marker=dict(color='seagreen'), name='Profit'), row=1, col=1)
    fig_tf.add_trace(go.Bar(x=flop_10['State'], y=flop_10['Profit'], marker=dict(color='crimson'), name='Pertes'), row=1, col=2)
    fig_tf.update_layout(showlegend=False)
    st.plotly_chart(fig_tf, width='stretch')

with tab2:
    st.subheader("Impact des Catégories sur la Marge")
    
    df_subcat = filtered_df.groupby(['Category', 'Sub-Category'], as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'})
    df_subcat = df_subcat.sort_values(by='Profit', ascending=True)

    fig_subcat = px.bar(
        df_subcat, y='Sub-Category', x='Profit', orientation='h',
        color='Discount', color_continuous_scale='Reds', text_auto='$.2s',
        hover_data={'Category': True, 'Sales': ':$,.0f', 'Discount': ':.1%'},
        title="Profit par Sous-Catégorie"
    )
    fig_subcat.update_layout(height=600)
    st.plotly_chart(fig_subcat, width='stretch')

with tab3:
    st.subheader("Analyse du Point Mort")

    df_discount = filtered_df.groupby('Discount', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Order ID': 'count'})
    df_discount['Marge Bénéficiaire'] = df_discount.apply(lambda row: row['Profit'] / row['Sales'] if row['Sales'] > 0 else 0, axis=1)

    fig_be = go.Figure()
    fig_be.add_trace(go.Scatter(
        x=df_discount['Discount'], y=df_discount['Marge Bénéficiaire'],
        mode='lines+markers', name='Marge Nette',
        line=dict(color='royalblue', width=3),
        marker=dict(size=8, color=df_discount['Marge Bénéficiaire'], colorscale='RdYlGn', showscale=False)
    ))
    fig_be.add_hline(y=0, line_dash="dash", line_color="red", annotation_text=" SEUIL DE RENTABILITÉ (0%)")
    fig_be.update_layout(
        title="Évolution de la Marge Nette selon le Taux de Remise",
        xaxis=dict(title='Taux de Remise', tickformat='.0%'),
        yaxis=dict(title='Marge Bénéficiaire Nette', tickformat='.0%')
    )
    st.plotly_chart(fig_be, width='stretch')

with tab4:
    st.subheader("Valeur par Segment Client")

    df_segment = filtered_df.groupby('Segment', as_index=False).agg({
        'Sales': 'sum', 
        'Profit': 'sum', 
        'Customer ID': 'nunique',
        'Order ID': 'nunique'
    })
    
    df_segment['Profit_par_Client'] = df_segment.apply(lambda r: r['Profit'] / r['Customer ID'] if r['Customer ID'] > 0 else 0, axis=1)
    df_segment['Profit_par_Commande'] = df_segment.apply(lambda r: r['Profit'] / r['Order ID'] if r['Order ID'] > 0 else 0, axis=1)
    
    df_segment = df_segment.sort_values(by='Profit_par_Client', ascending=False)

    col_client, col_order = st.columns(2)
    
    with col_client:
        fig_seg_client = px.bar(
            df_segment, x='Segment', y='Profit_par_Client',
            title="Profit Cumulé / Client (LTV)",
            text_auto='$.0f', color='Segment', color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_seg_client.update_layout(yaxis_title="Profit / Client ($)", showlegend=False)
        st.plotly_chart(fig_seg_client, width='stretch')

    with col_order:
        fig_seg_order = px.bar(
            df_segment, x='Segment', y='Profit_par_Commande',
            title="Profit / Commande",
            text_auto='$.1f', color='Segment', color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_seg_order.update_layout(yaxis_title="Profit / Commande ($)", showlegend=False)
        st.plotly_chart(fig_seg_order, width='stretch')
