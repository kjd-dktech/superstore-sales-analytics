# Copyright (c) 2026 Kodjo Jean DEGBEVI. Distribué sous licence CC BY-NC-SA 4.0.
#=============================================================================
# Application Streamlit pour l'analyse stratégique et prédictive du Superstore
#=============================================================================

import streamlit as st
from streamlit import iframe as stif

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
from pathlib import Path
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import time
import hashlib
import base64

# ==========================================
# CONFIGURATION, HISTORIQUE & MÉMOIRE
# ==========================================
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CURRENT_FILE_DIR = Path(__file__).resolve().parent
data_path = CURRENT_FILE_DIR / "superstore_processed.csv"

st.set_page_config(page_title="Superstore Analytics", page_icon="📊", layout="wide")

history_dir = CURRENT_FILE_DIR / "logs"
try:
    os.makedirs(history_dir, exist_ok=True)
except Exception:
    pass

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

session_id = st.session_state['session_id']
sim_history_file = history_dir / f"sim_history_{session_id}.csv"
batch_history_file = history_dir / f"batch_history_{session_id}.csv"

if 'sim_history' not in st.session_state:
    if sim_history_file.exists():
        try:
            st.session_state['sim_history'] = pd.read_csv(sim_history_file).to_dict('records')
        except:
            st.session_state['sim_history'] = []
    else:
        st.session_state['sim_history'] = []

if 'batch_history' not in st.session_state:
    if batch_history_file.exists():
        try:
            st.session_state['batch_history'] = pd.read_csv(batch_history_file).to_dict('records')
        except:
            st.session_state['batch_history'] = []
    else:
        st.session_state['batch_history'] = []

STREAMLIT_COOKIE_SECRET = os.getenv("STREAMLIT_COOKIE_SECRET", "super_secret_par_defaut")

_cookie_hash = hashlib.sha256(STREAMLIT_COOKIE_SECRET.encode('utf-8')).digest()
FERNET_KEY = base64.urlsafe_b64encode(_cookie_hash)
cipher = Fernet(FERNET_KEY)

def get_cookie(cookie_name):
    if hasattr(st, "context") and hasattr(st.context, "cookies"):
        return st.context.cookies.get(cookie_name)
    return None

def set_cookie_js(name, value, max_age_days=7):
    stif(f"""
        <script>
            document.cookie = "{name}={value}; max-age={max_age_days*86400}; path=/; SameSite=Strict; Secure";
        </script>
    """, height=1)

def clear_cookie_js(name):
    stif(f"""
        <script>
            document.cookie = "{name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Strict; Secure";
        </script>
    """, height=1)

if 'api_key_val' not in st.session_state:
    encrypted_cookie = get_cookie('app_key')
    if encrypted_cookie:
        try:
            st.session_state['api_key_val'] = cipher.decrypt(encrypted_cookie.encode()).decode()
        except:
            st.session_state['api_key_val'] = ""
    else:
        st.session_state['api_key_val'] = ""

def save_api_key():
    val = st.session_state.get('api_input', '')
    st.session_state['api_key_val'] = val
    if val:
        encrypted_val = cipher.encrypt(val.encode()).decode()
        set_cookie_js('app_key', encrypted_val)
    else:
        clear_cookie_js('app_key')

def save_history_to_disk(history_type):
    try:
        if history_type == 'sim':
            pd.DataFrame(st.session_state['sim_history']).to_csv(sim_history_file, index=False)
        elif history_type == 'batch':
            pd.DataFrame(st.session_state['batch_history']).to_csv(batch_history_file, index=False)
    except Exception:
        pass

def read_data_robust(file_obj, file_name=""):
    if file_name.endswith('.json'):
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_json(file_obj)
    elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_excel(file_obj)
        
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'latin1', 'utf-16', 'utf-8-sig', 'mac_roman']
    for enc in encodings:
        try:
            if hasattr(file_obj, 'seek'): file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Impossible de décoder le fichier CSV. Encodages tentés: {', '.join(encodings)}")

def get_us_state_abbrev():
    return {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        'District of Columbia': 'DC'
    }

@st.cache_data(ttl=60, show_spinner=False)
def check_api_status():
    try:
        res = requests.get(f"{API_URL}/", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return {
                "online": True,
                "model_loaded": data.get("model_loaded", False)
            }
    except:
        pass
    
    return {
        "online": False,
        "model_loaded": False
    }

@st.cache_data(ttl=300, show_spinner=False)
def validate_api_key(key: str):
    if not key:
        return {"valid": False, "reason": "missing"}
    
    try:
        #payload = {
        #    'Sales': 0,
        #    'Discount': 0,
        #    'Sub_Category': 'Chairs',
        #    'Region': 'West',
        #    'Segment': 'Consumer'
        #}

        res = requests.get(
            f"{API_URL}/auth/check",
            #json=payload,
            headers={"X-API-KEY": key},
            timeout=6
        )

        if res.status_code == 200:
            return {"valid": True, "reason": "ok"}
        elif res.status_code == 403:
            return {"valid": False, "reason": "invalid"}
        elif res.status_code == 503:
            return {"valid": False, "reason": "model_down"}
        else:
            return {"valid": False, "reason": "error"}

    except:
        return {"valid": False, "reason": "unreachable"}

api_status = check_api_status()
api_online = api_status["online"]
model_loaded = api_status["model_loaded"]

master_key = os.getenv("MASTER_API_KEY", "")
current_api_key = st.session_state.get('api_key_val', '')

using_master = False
key_status = None

# Priorité à la disponibilité de l'API
if api_online:
    
    if master_key:
        key_status = validate_api_key(master_key)
        if key_status["valid"]:
            current_api_key = master_key
            using_master = True
    
    if not using_master:
        key_status = validate_api_key(current_api_key)

@st.cache_data
def load_data():
    df = read_data_robust(data_path, "superstore_processed.csv")
        
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Order Year'] = df['Order Date'].dt.year
    return df

df = load_data()

# ==========================================
# SIDEBAR : API & FILTRAGE
# ==========================================
# --- Config API ---

st.sidebar.header("Configuration API")

api_key = current_api_key or ""
if not api_online:
    st.sidebar.warning("⚠️ API injoignable. Vérifiez votre connexion.")
    
elif not model_loaded:
    st.sidebar.warning("⚠️ Modèle non chargé côté serveur.")
    
elif using_master:
    api_key = current_api_key
    st.sidebar.success("✅ Application connectée.")
else:
    if current_api_key and key_status and key_status["valid"]:
        api_key = current_api_key
        st.sidebar.success("✅ Connecté à l'API")
        if not using_master:
            if st.sidebar.button("Déconnecter / Changer la clé"):
                st.session_state['api_key_val'] = ""
                save_api_key()
                api_key = ""
                st.rerun()
    else:
        api_key = st.sidebar.text_input(
            "Clé API",
            type="password",
            key="api_input",
            value=current_api_key,
            on_change=save_api_key,
            help="Appuyez sur Entrée pour valider.[Obtenir une clé API](https://kjd-dktech-superstore-api.hf.space/developer)."
        )

        if not current_api_key:
            st.sidebar.info("ℹ️ Entrez votre clé API.")
        
        elif key_status["reason"] == "invalid":
            st.sidebar.error("❌ Clé API invalide.")
        
        elif key_status["reason"] == "model_down":
            st.sidebar.warning("⚠️ Modèle indisponible.")
        
        elif key_status["reason"] == "unreachable":
            st.sidebar.warning("⚠️ Serveur injoignable.")
        
        elif key_status["reason"] == "error":
            st.sidebar.error("❌ Erreur lors de la validation de la clé API.")

# --- Config Filtres ---
st.sidebar.header("Filtres")

selected_years = st.sidebar.multiselect(
    "Années", 
    options=sorted(df['Order Year'].unique()) if 'Order Year' in df.columns else [], 
    default=sorted(df['Order Year'].unique()) if 'Order Year' in df.columns else []
)

selected_regions = st.sidebar.multiselect(
    "Régions", 
    options=sorted(df['Region'].unique()), 
    default=sorted(df['Region'].unique())
)

selected_categories = st.sidebar.multiselect(
    "Catégories", 
    options=sorted(df['Category'].unique()) if 'Category' in df.columns else [], 
    default=sorted(df['Category'].unique()) if 'Category' in df.columns else []
)

if 'Order Year' in df.columns and not selected_years: selected_years = df['Order Year'].unique()
if not selected_regions: selected_regions = df['Region'].unique()
if 'Category' in df.columns and not selected_categories: selected_categories = df['Category'].unique()

mask = df['Region'].isin(selected_regions)
if 'Order Year' in df.columns:
    mask &= df['Order Year'].isin(selected_years)
if 'Category' in df.columns:
    mask &= df['Category'].isin(selected_categories)
    
filtered_df = df[mask]

# ==========================================
# HEADER & KPIS
# ==========================================
st.title("📊 Dashboard - Superstore")

col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df['Sales'].sum() if 'Sales' in filtered_df.columns else 0
total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
marge_globale = total_profit / total_sales if total_sales > 0 else 0
total_customers = filtered_df['Customer ID'].nunique() if 'Customer ID' in filtered_df.columns else 0

# Amplitude temporelle
if not filtered_df.empty and 'Order Date' in filtered_df.columns:
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

tabs_names = [
    "� Recommandations",
    "📍 Géographie", 
    "📦 Rentabilité Produit", 
    "💸 Point Mort", 
    "👥 Valeur Client",
    "🤖 Modélisation",
    "🕒 Historique (Logs)"
]

tab_reco, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs_names)

st.markdown("---")

# ==========================================
# VUES DES ONGLETS
# ==========================================
with tab_reco:
    #st.header("Résumé Exécutif & Recommandations Stratégiques")
    
    st.markdown("""
    
    #### 1. Instauration d'un Plafond de Remise (Hard Cap)
    **Constat :** L'analyse du point mort démontre que toute remise supérieure à 20% entraîne quasi systématiquement une vente à perte, particulièrement sur les produits dont la structure de coût est rigide.
    
    > **Recommandation :** Implémenter une règle métier stricte dans le système de vente bloquant techniquement toute remise au-delà de 20%. <br> Les exceptions doivent être limitées aux produits à très haute marge brute (ex: Classeurs/Binders) et soumises à validation managériale.
    
    ---
    
    #### 2. Redressement des États "Trous Noirs"
    **Constat :** Seuls quatre États (Texas, Ohio, Pennsylvanie, Illinois) cumulent plus de 70 000 $ de pertes nettes en raison de remises moyennes insoutenables dépassant les 30%, contre seulement 11% dans la région Ouest.
    
    > **Recommandation :** Aligner d'urgence la politique commerciale de ces zones sur le modèle de discipline tarifaire de la Californie ou de New York. <br> Il est préférable de sacrifier une partie du volume de ventes au profit d'un retour immédiat à la rentabilité par la réduction drastique des promotions. <br> En parallèle, un audit des opérations locales (Supply Chain régionale) doit être mené pour identifier si ces remises massives servent à compenser des coûts de distribution ou de stockage structurellement défaillants dans ces quatre États.
    
    ---
    
    #### 3. Rationalisation de l'Offre Tables & Machines
    **Constat :** Contrairement aux fournitures de bureau, les catégories "Tables" et "Machines" ne supportent pas les stratégies de remise agressive. Elles sont actuellement les principales contributrices au déficit opérationnel.
    
    > **Recommandation :** Cesser d'utiliser ces catégories comme produits d'appel promotionnels. <br> Elles doivent être repositionnées comme des produits de destination vendus à prix plein ou avec des remises symboliques, ou être retirées du catalogue dans les régions les moins rentables. <br> Un audit ciblé sur les coûts de fret et d'entreposage de ces biens encombrants s'impose pour valider s'il est logistiquement viable de continuer à les distribuer sur l'ensemble du territoire.
    
    ---
    
    #### 4. Pivot Marketing vers le Segment Home Office
    **Constat :** Le segment Home Office (TPE/Indépendants) s'avère 24% plus rentable que le segment Consumer (Particuliers) pour un coût d'acquisition client (CAC) similaire.
    
    > **Recommandation :** Réallouer prioritairement le budget marketing vers le segment Home Office. <br> Concevoir des offres d'équipement technologique (catégorie à forte marge) spécifiquement packagées pour les travailleurs à domicile afin de maximiser le retour sur investissement des campagnes.
    """, unsafe_allow_html=True)

with tab1:
    st.subheader("Cartographie de la Rentabilité")
    if 'State' in filtered_df.columns and 'Sales' in filtered_df.columns and 'Profit' in filtered_df.columns and 'Discount' in filtered_df.columns:
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
    if 'Category' in filtered_df.columns and 'Sub-Category' in filtered_df.columns:
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
    if 'Discount' in filtered_df.columns and 'Sales' in filtered_df.columns and 'Profit' in filtered_df.columns:
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
    if 'Segment' in filtered_df.columns and 'Customer ID' in filtered_df.columns:
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
                title="Profit Cumulé / Client",
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

with tab5:
    st.subheader("Modélisation de la Rentabilité")
    
    if not api_online:
        st.warning("⚠️ L'API est introuvable. Vérifiez votre connexion ou le serveur FastAPI.")
    
    elif not model_loaded:
        st.warning("⚠️ Le modèle de prédiction n'est pas disponible côté serveur.")
    
    elif not api_key or (key_status and key_status["reason"] == "missing"):
        st.info("ℹ️ Veuillez entrer votre clé API dans le menu latéral.")
    
    elif not key_status:
        st.error("❌ Clé API invalide ou non reconnue.")
    
    elif key_status["reason"] == "invalid":
        st.error("❌ Clé API invalide.")
    
    elif key_status["reason"] == "unreachable":
        st.warning("⚠️ Impossible de vérifier la clé (serveur injoignable).")
    
    elif key_status["reason"] == "model_down":
        st.warning("⚠️ Modèle de prédiction indisponible.")
    
    elif key_status["reason"] == "error":
        st.error("❌ Erreur lors de la validation de la clé API.")
    
    else:
        tab_sim, tab_batch = st.tabs(["🎯 Simulateur What-If", "📂 Traitement par Lot"])

        with tab_sim:
            st.markdown("### Simulateur What-If")
            with st.form("whatif_form"):
                sim_sales = st.number_input("Montant de la vente ($)", min_value=0.0, value=500.0, step=50.0)
                sim_discount = st.slider("Taux de remise (%)", min_value=0.0, max_value=0.8, value=0.0, step=0.01)
                
                cats_available = df['Sub-Category'].unique() if 'Sub-Category' in df.columns else ['Chairs', 'Phones', 'Storage']
                sim_subcat = st.selectbox("Sous-Catégorie", cats_available)
                
                regs_available = df['Region'].unique() if 'Region' in df.columns else ['West', 'East', 'Central', 'South']
                sim_region = st.selectbox("Région", regs_available)
                
                segs_available = df['Segment'].unique() if 'Segment' in df.columns else ['Consumer', 'Corporate', 'Home Office']
                sim_segment = st.selectbox("Segment Client", segs_available)
                
                submit_sim = st.form_submit_button("Calculer le Profit Estimé")
                
            if submit_sim:
                payload = {
                    'Sales': sim_sales, 'Discount': sim_discount, 
                    'Sub_Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment
                }
                
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload, headers={"X-API-KEY": api_key})
                    
                    if res.status_code == 200:
                        pred_profit = res.json()["predicted_profit"]
                        marge = pred_profit / sim_sales if sim_sales > 0 else 0
                        
                        st.success("✅ **Prédiction Réalisée avec Succès !**")
                        
                        res_col1, res_col2 = st.columns(2)
                        res_col1.metric(label="Profit Net Estimé", value=f"{pred_profit:,.2f} $", delta=f"{marge:.2%} de marge", delta_color="normal")
                        res_col2.metric(label="Chiffre d'Affaires Simulé", value=f"{sim_sales:,.2f} $", delta=f"Remise: {sim_discount:.0%}", delta_color="inverse")

                        st.markdown("#### Variation du profit estimé selon la remise")
                        d_range = np.linspace(0, 0.8, 20)
                        
                        batch_payload = {"records": [
                            {'Sales': sim_sales, 'Discount': float(d), 'Sub_Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment}
                            for d in d_range
                        ]}
                        
                        res_batch = requests.post(f"{API_URL}/predict_batch", json=batch_payload, headers={"X-API-KEY": api_key})
                        
                        if res_batch.status_code == 200:
                            sim_df = pd.DataFrame([{
                                'Sales': sim_sales, 'Discount': d,
                                'Sub-Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment
                            } for d in d_range])
                            sim_df['Profit Estimé'] = res_batch.json()["predictions"]
                            fig_sim = px.line(sim_df, x='Discount', y='Profit Estimé', markers=True, labels={'Discount': 'Taux de Remise', 'Profit Estimé': 'Profit Estimé ($)'})
                            fig_sim.add_hline(y=0, line_dash="dash", line_color="red")
                            fig_sim.add_vline(x=sim_discount, line_dash="dot", line_color="green", annotation_text="Remise Actuelle")
                            fig_sim.update_layout(xaxis=dict(tickformat='.0%'))
                            st.plotly_chart(fig_sim, width='stretch')
                        else:
                            st.warning(f"Erreur API pour la variation ({res_batch.status_code}): {res_batch.text}")

                        st.session_state['sim_history'].append({
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Sales ($)": sim_sales,
                            "Discount (%)": sim_discount,
                            "Sub-Category": sim_subcat,
                            "Region": sim_region,
                            "Profit Estimé ($)": round(pred_profit, 2)
                        })
                        save_history_to_disk('sim')
                    elif res.status_code in [401, 403]:
                        st.error("🔑 Votre clé d'accès n'est plus valide ou a été révoquée. Déconnexion requise...")
                        st.session_state['api_key_val'] = ""
                        clear_cookie_js('app_key')
                        time.sleep(2)
                        st.rerun()
                    elif res.status_code == 429:
                        st.warning("⏳ Limite de requêtes atteinte. Veuillez patienter un instant.")
                    else:
                        st.error(f"Accès Refusé / Erreur API ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Erreur de communication avec l'API : {e}")

        with tab_batch:
            st.markdown("### Traitement par Lot (Batch CSV, XLS, JSON)")
            uploaded_file = st.file_uploader("Chargez vos données transactionnelles", type=["csv", "xls", "xlsx", "json"])
            
            if uploaded_file is not None:
                try:
                    # Lecture : CSV multiprocess, JSON, ou Excel
                    batch_df = read_data_robust(uploaded_file, uploaded_file.name)
                        
                    st.success(f"{len(batch_df)} lignes chargées avec succès.")
                    
                    required_cols = ['Sales', 'Discount', 'Sub-Category', 'Region', 'Segment']
                    if all(c in batch_df.columns for c in required_cols):
                        
                        api_records = batch_df[required_cols].rename(columns={"Sub-Category": "Sub_Category"}).to_dict(orient='records')
                        
                        try:
                            res_batch_file = requests.post(f"{API_URL}/predict_batch", json={"records": api_records}, headers={"X-API-KEY": api_key})
                            
                            if res_batch_file.status_code == 200:
                                batch_df['Predicted_Profit'] = res_batch_file.json()["predictions"]
                                
                                st.dataframe(batch_df[['Sub-Category', 'Sales', 'Discount', 'Predicted_Profit']].head())

                                st.session_state['batch_history'].append({
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Fichier": uploaded_file.name,
                                    "Lignes": len(batch_df),
                                    "Statut": "Succès"
                                })
                                save_history_to_disk('batch')

                                st.markdown("📥 **Télécharger les résultats**")
                                
                                # Export JSON
                                json_data = batch_df.to_json(orient='records').encode('utf-8')
                                st.download_button("Télécharger JSON", data=json_data, file_name="predictions_batch.json", mime="application/json")
                                
                                # Export CSV
                                csv_data = batch_df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                                st.download_button("Télécharger CSV", data=csv_data, file_name="predictions_batch.csv", mime="text/csv")
                                
                                # Export Excel
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    batch_df.to_excel(writer, index=False, sheet_name='Prédictions')
                                st.download_button("Télécharger Excel", data=buffer.getvalue(), file_name="predictions_batch.xlsx", mime="application/vnd.ms-excel")
                            elif res_batch_file.status_code in [401, 403]:
                                st.error("🔑 Votre clé d'accès n'est plus valide ou a été révoquée. Déconnexion requise...")
                                st.session_state['api_key_val'] = ""
                                clear_cookie_js('app_key')
                                time.sleep(2)
                                st.rerun()
                            elif res_batch_file.status_code == 429:
                                st.warning("⏳ Limite de requêtes atteinte. Veuillez patienter un instant.")
                            else:
                                st.error(f"Erreur API ({res_batch_file.status_code}) : {res_batch_file.text}")
                        except Exception as api_err:
                            st.error(f"Erreur de communication avec l'API : {api_err}")
                            
                    else:
                        st.error(f"Le fichier doit impérativement contenir les colonnes : {', '.join(required_cols)}")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier : {e}")

with tab6:
    st.subheader("Historique de Session")
    
    st.markdown("#### Simulations Unitaires")
    if st.session_state['sim_history']:
        st.dataframe(pd.DataFrame(st.session_state['sim_history']), width='stretch')
    else:
        st.info("Aucune simulation unitaire réalisée dans cette session.")
        
    st.markdown("#### Imports")
    if st.session_state['batch_history']:
        st.dataframe(pd.DataFrame(st.session_state['batch_history']), width='stretch')
    else:
        st.info("Aucun import par lot validé dans cette session.")
        
    if st.button("🗑️ Vider l'historique"):
        st.session_state['sim_history'] = []
        st.session_state['batch_history'] = []
        save_history_to_disk('sim')
        save_history_to_disk('batch')
        st.rerun()
