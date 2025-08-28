# ========================================================================
# -------------------- BLOQUE 1: IMPORTS Y CONFIGURACIÓN GENERAL --------------------
# ========================================================================
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pandas.api.types import is_numeric_dtype
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import rankdata

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from utils.utils_visuals import (
    crear_pizza_chart
)

import re
import unicodedata
# Importa tu helper (el que ya tienes en Inicio.py)
from Inicio import get_theme_type  # ajusta el import según tu estructura real

# --- Sanitizador simple para textos de hover/etiquetas (evita caracteres de control) ---
_re_bad = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def _sanitize_text(val) -> str:
    try:
        return _re_bad.sub("", str(val))
    except Exception:
        return str(val)

# --- Clave de orden A→Z (ignora mayúsculas y tildes) ---
def _sort_key_az(s: str) -> str:
    try:
        s = str(s)
    except Exception:
        s = ""
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()

# --- Sanitizadores de colores HEX (para arrays de Plotly) ---
_hex_ctrl = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029\ufeff]")
_hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")

def _clean_hex_color(val) -> str | None:
    """Normaliza entradas del tipo '#RRGGBB' o 'RRGGBB'. Quita caracteres de control,
    espacios y comillas. Devuelve '#RRGGBB' o None si no es válido."""
    if val is None:
        return None
    s = str(val)
    s = _hex_ctrl.sub("", s)   # elimina controles y BOM
    s = s.strip().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    s = s.replace('"', '').replace("'", "")
    if not s:
        return None
    if s.startswith('#'):
        s = s[1:]
    # dejar sólo caracteres hex
    s = ''.join(ch for ch in s if ch in '0123456789abcdefABCDEF')
    if len(s) == 6:
        return '#' + s.upper()
    return None


from config import DATA_PARQUET, METRICAS_EXCEL
try:
    from utils.utils_data import (
        cargar_datos, cargar_metricas,
        aplicar_metricas_personalizadas, diminutivos_pos, formatear_valor, 
        metricas_fisicas, metricas_centros, metricas_construccion_general, metricas_construccion_ofensiva,
        metricas_ofensivas, metricas_balon_parado, metricas_portero, tipos_default_por_posicion, 
        metricas_default_por_posicion, obtener_percentiles
    )
except Exception:
    from utils.utils_data import (
        cargar_datos, cargar_metricas,
        aplicar_metricas_personalizadas, diminutivos_pos
    )
    def formatear_valor(metric_name, v):
        try:
            return f"{v:.2f}"
        except Exception:
            return str(v)

# Configuración general de la página
st.set_page_config(
    page_title="Dashboard general",
    layout="wide",
    page_icon="📊"
)

# -- Ajuste visual: reducir tamaño de texto en MultiSelect --
st.markdown(
    """
    <style>
    /* Texto dentro del desplegable y los elementos seleccionados */
    [data-testid="stMultiSelect"] div[data-baseweb="select"] * { font-size: 0.85rem !important; }
    /* Etiqueta del multiselect */
    [data-testid="stMultiSelect"] label p { font-size: 0.85rem !important; }
    /* Chips/etiquetas seleccionadas */
    [data-baseweb="tag"] { font-size: 0.80rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Colores dependientes del tema ===
def _theme_colors():
    base = st.get_option("theme.base")
    if base == "dark":
        return {
            "bg_color": "#0e1117",
            "ejes_color": "#e0e0e0",
            "font_color": "#e0e0e0",
            "linea_25": "#ff4b4b",  # rojo
            "linea_75": "#2ecc71",  # verde
        }
    else:
        return {
            "bg_color": "#ffffff",
            "ejes_color": "#333333",
            "font_color": "#333333",
            "linea_25": "#d62728",
            "linea_75": "#2ca02c",
        }
_c = _theme_colors()
bg_color   = _c["bg_color"]
ejes_color = _c["ejes_color"]
font_color = _c["font_color"]
linea_25   = _c["linea_25"]
linea_75   = _c["linea_75"]

# ========================================================================
# -------------------- BLOQUE 2: PREPARACIÓN Y CARGA DE DATOS --------------------
# ========================================================================
@st.cache_data(show_spinner="Cargando y preparando datos...")
def preparar_datos_dashboard():
    # 1) Cargar datos base
    df = cargar_datos(DATA_PARQUET)
    # 2) Cargar catálogo de métricas
    df_metricas = cargar_metricas(METRICAS_EXCEL)

    # 4) Tipos: asegurar numéricos en columnas críticas sin forzar todo el DF
    for col in ["Edad", "M90s_jugados", "Minutos_jugados"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Filtrar registros válidos básicos
    if "Nombre_transfermarket" in df.columns:
        df = df[df["Nombre_transfermarket"].notna()].copy()

    return df, df_metricas

df, df_metricas = preparar_datos_dashboard()
df_all = df.copy()

# --- Listas completas (sin filtros): equipos y jugadores ---
@st.cache_data(show_spinner=False)
def obtener_catalogos_globales(parquet_path: str):
    """Devuelve (equipos_catalogo, jugadores_df, df_all) sin respetar filtros previos.
    - equipos_catalogo: lista ordenada (Equipo_data_full si es posible)
    - jugadores_df: DataFrame con columnas ['ID', 'label', 'Equipo_data_full']
    - df_all: DataFrame completo del parquet
    """
    df_all = cargar_datos(parquet_path)
    # Construir Equipo_data_full si no existe
    if "Equipo_data_full" not in df_all.columns:
        if {"Equipo_data", "Pais_diminutivo"}.issubset(df_all.columns):
            df_all["Equipo_data_full"] = df_all["Equipo_data"].astype(str) + " " + df_all["Pais_diminutivo"].astype(str)
        elif "Equipo_data" in df_all.columns:
            df_all["Equipo_data_full"] = df_all["Equipo_data"].astype(str)
        else:
            df_all["Equipo_data_full"] = ""
    # Etiqueta de jugador
    if "ID_Display" in df_all.columns:
        labels = df_all["ID_Display"].astype(str)
    elif "Nombre_transfermarket" in df_all.columns:
        labels = df_all["Nombre_transfermarket"].astype(str)
    else:
        labels = df_all.get("ID", pd.Series(range(len(df_all)))).astype(str)
    # ID de jugador
    if "ID" not in df_all.columns:
        df_all["ID"] = pd.factorize(labels)[0]
    jugadores_df = pd.DataFrame({
        "ID": df_all["ID"],
        "label": labels,
        "Equipo_data_full": df_all["Equipo_data_full"].astype(str)
    }).dropna(subset=["label"]).drop_duplicates()
    equipos_catalogo = sorted(
        jugadores_df["Equipo_data_full"].dropna().unique().tolist(),
        key=_sort_key_az
    )
    return equipos_catalogo, jugadores_df, df_all

# Fecha de actualización del dataset
parquet_path = DATA_PARQUET
if os.path.exists(parquet_path):
    ts = os.path.getmtime(parquet_path)
    fecha_actualizacion = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
else:
    fecha_actualizacion = "desconocida"


# ========================================================================
# -------------------- BLOQUE 3: FILTROS EN SIDEBAR Y EXPANDER --------------------
# ========================================================================
st.sidebar.markdown("## ⚙️ Filtros para la muestra")

 # --- Temporada ---
temporadas_disponibles = sorted(df["Temporada"].dropna().unique(), reverse=True) if "Temporada" in df.columns else []
if not temporadas_disponibles:
    st.error("No hay temporadas disponibles en los datos.")
    st.stop()

default_temporada = next((t for t in temporadas_disponibles if "2025" in str(t)), temporadas_disponibles[0])
# --- PATCH: override temporada desde session_state ---
override_temp = st.session_state.get('1v1_sync_temporada')
if override_temp in temporadas_disponibles:
    default_temporada = override_temp
idx_temp = temporadas_disponibles.index(default_temporada) if default_temporada in temporadas_disponibles else 0
temporada = st.sidebar.selectbox(
    "Temporada",
    temporadas_disponibles,
    index=idx_temp
)
df_temp = df[df["Temporada"] == temporada].copy()

# --- País del equipo ---
paises_opciones = sorted(df_temp["Pais"].dropna().unique()) if "Pais" in df_temp.columns else []
default_pais = "Peru" if "Peru" in paises_opciones else (paises_opciones[0] if paises_opciones else None)
# --- PATCH: override paises desde session_state ---
override_paises = st.session_state.get('1v1_sync_paises')
paises_sel = st.sidebar.multiselect(
    'País del equipo',
    paises_opciones,
    default=(override_paises if override_paises else ([default_pais] if default_pais else []))
)

# --- Torneo ---
if paises_sel:
    torneos_opciones = sorted(
        df_temp.loc[df_temp["Pais"].isin(paises_sel), "Torneo"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
else:
    torneos_opciones = sorted(
        df_temp["Torneo"].dropna().astype(str).str.strip().unique()
    )

# --- PATCH: override torneos desde session_state ---
override_torneos = st.session_state.get('1v1_sync_torneos')
torneos_sel = st.sidebar.multiselect(
    'Torneo',
    torneos_opciones,
    default=(override_torneos if override_torneos else torneos_opciones)
)

# DF filtrado preliminar (pais/torneo) – hacemos copy para evitar SettingWithCopy
mask_base = df_temp["Pais"].isin(paises_sel) & df_temp["Torneo"].isin(torneos_sel)
df_filtros = df_temp[mask_base].copy()


# -------------------- FILTROS RESTANTES EN SIDEBAR --------------------
with st.sidebar:

    if df_filtros.empty:
        st.warning("No hay datos para los filtros actuales de País/Torneo.")
        st.stop()

    # Minutos / M90s / Edad
    min_jugados_max = int(df_filtros["Minutos_jugados"].max()) if "Minutos_jugados" in df_filtros.columns else 0
    min_jugados = st.slider("Minutos jugados (mínimo)", 0, max(min_jugados_max, 0), min(300, max(min_jugados_max, 0)), step=50)

    max_m90 = int(df_filtros["M90s_jugados"].max()) if "M90s_jugados" in df_filtros.columns else 0
    min_m90s = st.slider("Partidos completos jugados (M90s)", 0, max(max_m90, 0), min(3, max(max_m90, 0)), step=1)

    if "Edad" in df_filtros.columns:
        edad_min = int(df_filtros["Edad"].min()); edad_max = int(df_filtros["Edad"].max())
        edad_range = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
    else:
        edad_range = (0, 100)

    # Posición general (visual: POR/DEF/MED/DEL en ese orden)
    abbr_map = {
        "Portero": "POR",
        "Defensa": "DEF",
        "Mediocampista": "MED",
        "Delantero": "DEL",
    }
    rev_abbr = {v: k for k, v in abbr_map.items()}

    posiciones_gen_raw = (
        df_filtros["Posicion_general"].dropna().unique().tolist()
        if "Posicion_general" in df_filtros.columns else []
    )
    # Orden forzado POR -> DEF -> MED -> DEL, manteniendo sólo presentes
    orden_full = ["Portero", "Defensa", "Mediocampista", "Delantero"]
    posiciones_gen = [p for p in orden_full if p in posiciones_gen_raw]

    # Default: todas menos Portero si existe, salvo override desde Step 2 (Jugador 1)
    default_pos_gen = [p for p in posiciones_gen if p != "Portero"] if posiciones_gen else []
    override_pos_gen = st.session_state.get('1v1_sync_pos_gen')
    if override_pos_gen and override_pos_gen in posiciones_gen:
        default_pos_gen = [override_pos_gen]

    # Opciones visuales (abreviaturas) y defaults visuales
    opciones_vis = [abbr_map[p] for p in posiciones_gen]
    default_vis = [abbr_map[p] for p in default_pos_gen]

    # Sembrar selección deseada en Session State (sin pasar `default` al widget)
    desired_vis = st.session_state.get("pos_gen_seg_1v1")
    # Si hay override desde Step 2, forzar ese valor (en abreviatura)
    if override_pos_gen and override_pos_gen in abbr_map:
        desired_vis = [abbr_map[override_pos_gen]]
    if desired_vis is None:
        desired_vis = default_vis
    # Guardamos ANTES de instanciar el widget; no pasamos `default` para evitar la advertencia
    st.session_state["pos_gen_seg_1v1"] = desired_vis

    _seg = getattr(st, "segmented_control", None)
    if callable(_seg):
        try:
            sel_vis = _seg("Posición general", opciones_vis, selection_mode="multi", key="pos_gen_seg_1v1")
        except Exception:
            sel_vis = _seg("Posición general", opciones_vis, key="pos_gen_seg_1v1")
    else:
        sel_vis = st.multiselect("Posición general", opciones_vis, key="pos_gen_seg_1v1")

    if isinstance(sel_vis, str):
        sel_vis = [sel_vis]
    # Convertir abreviaturas a labels reales usadas en el dataset
    pos_gen_sel = [rev_abbr.get(lbl, lbl) for lbl in sel_vis]

    # Posición detallada
    if pos_gen_sel and "Posicion_detallada" in df_filtros.columns:
        _raw_det = df_filtros[df_filtros["Posicion_general"].isin(pos_gen_sel)]["Posicion_detallada"].dropna().astype(str)
        _bad = {"nan", "none", "null", "", "na"}
        _raw_det = _raw_det[~_raw_det.str.strip().str.lower().isin(_bad)]
        pos_det_opciones = sorted(_raw_det.unique())
    else:
        pos_det_opciones = []
    pos_det_sel = st.multiselect("Posición detallada", pos_det_opciones, default=pos_det_opciones)

    # Nacionalidad
    usar_nacionalidad_detallada = st.checkbox("Cambiar a nacionalidad detallada", value=False)
    if usar_nacionalidad_detallada:
        if "Nacionalidad" in df_filtros.columns:
            df_filtros["Nacionalidad"] = df_filtros["Nacionalidad"].astype(str)
            nacionalidades_detalladas = sorted(df_filtros["Nacionalidad"].dropna().unique())
            nac2_sel = st.multiselect("Nacionalidad", nacionalidades_detalladas, default=nacionalidades_detalladas)
        else:
            nac2_sel = []
    else:
        if "Nacionalidad_2" in df_filtros.columns:
            nacionalidades = sorted(df_filtros["Nacionalidad_2"].dropna().unique())
            _seg = getattr(st, "segmented_control", None)
            if callable(_seg):
                try:
                    nac2_sel = _seg("Nacionalidad", nacionalidades, selection_mode="multi", default=nacionalidades)
                except Exception:
                    nac2_sel = _seg("Nacionalidad", nacionalidades, default=nacionalidades)
                    if isinstance(nac2_sel, str):
                        nac2_sel = [nac2_sel]
            else:
                nac2_sel = st.multiselect("Nacionalidad", nacionalidades, default=nacionalidades)
        else:
            nac2_sel = []


# --- Filtro final aplicado sobre df_temp ---
if df_temp.empty:
    st.warning("No hay datos para la temporada seleccionada.")
    st.stop()

df = df_temp[
    (df_temp["Pais"].isin(paises_sel)) &
    (df_temp["Torneo"].isin(torneos_sel)) &
    (df_temp["Minutos_jugados"] >= min_jugados) &
    (df_temp["M90s_jugados"] >= min_m90s) &
    (df_temp["Edad"] >= edad_range[0]) & (df_temp["Edad"] <= edad_range[1]) &
    (df_temp["Posicion_general"].isin(pos_gen_sel)) &
    (df_temp["Posicion_detallada"].isin(pos_det_sel)) &
    (
        df_temp["Nacionalidad"].isin(nac2_sel)
        if usar_nacionalidad_detallada and "Nacionalidad" in df_temp.columns
        else df_temp["Nacionalidad_2"].isin(nac2_sel) if "Nacionalidad_2" in df_temp.columns else True
    )
].copy()

if df.empty:
    st.warning("⚠️ No hay jugadores que cumplan los filtros aplicados. Ajusta los filtros para continuar.")
    st.stop()


# Inicializar variables de session_state específicas de la página 1v1
if '1v1_step' not in st.session_state:
    st.session_state['1v1_step'] = 1
if '1v1_jugador_1_display' not in st.session_state:
    st.session_state['1v1_jugador_1_display'] = None
if '1v1_jugador_2_display' not in st.session_state:
    st.session_state['1v1_jugador_2_display'] = None
if '1v1_torneos_1' not in st.session_state:
    st.session_state['1v1_torneos_1'] = None
if '1v1_torneos_2' not in st.session_state:
    st.session_state['1v1_torneos_2'] = None

# -------------------- BLOQUE NUEVO (actualizado): CATÁLOGO DE JUGADORES (SIN FILTROS) --------------------
# IDs y labels para selección 1v1 usando TODO el dataset (sin respetar filtros)
if "ID_Display" not in df_all.columns and "Nombre_transfermarket" in df_all.columns:
    df_all["ID_Display"] = df_all["Nombre_transfermarket"].astype(str)
if "Equipo_data_full" not in df_all.columns and "Equipo_data" in df_all.columns:
    if "Pais_diminutivo" in df_all.columns:
        df_all["Equipo_data_full"] = df_all["Equipo_data"].astype(str) + " " + df_all["Pais_diminutivo"].astype(str)
    else:
        df_all["Equipo_data_full"] = df_all["Equipo_data"].astype(str)

ids_disponibles = df_all.dropna(subset=["ID", "ID_Display"]).copy()
ids_disponibles = ids_disponibles[["ID", "ID_Display", "Equipo_data_full", "Temporada"]].drop_duplicates()
ids_disponibles = ids_disponibles.sort_values("ID_Display", key=lambda s: s.map(_sort_key_az))

# --- Layout principal en 2 columnas (izquierda: pasos; derecha: ficha) ---
col_izq, col_der = st.columns([1,1])
col_der_placeholder = col_der.container()

## -------------------- FLUJO DE SELECCIÓN DE JUGADOR Y TORNEOS --------------------
# Funciones auxiliares para resetear selección (1 jugador)

def reset_jugador():
    st.session_state['1v1_step'] = 1
    st.session_state['1v1_jugador_1_display'] = None
    st.session_state['1v1_torneos_1'] = None


def reset_torneos():
    st.session_state['1v1_step'] = 2
    st.session_state['1v1_torneos_1'] = None

# Paso 1: Selección de JUGADOR BASE con expander y título dinámico
expander_title = "1️⃣ Confirmar jugador base"
with col_izq:
    with st.expander(expander_title, expanded=True):
        if st.session_state['1v1_step'] == 1 or st.session_state['1v1_jugador_1_display'] is None:
            with st.form("seleccion_jugador_base"):
                col1, col2 = st.columns([1,1])
                with col1:
                    jugador_1_display = st.selectbox(
                        "Jugador base",
                        sorted(ids_disponibles["ID_Display"]),
                        key="1v1_jugador_1_select"
                    )
                submit_jugador = st.form_submit_button("Confirmar jugador")
            if submit_jugador:
                st.session_state['1v1_jugador_1_display'] = jugador_1_display
                st.session_state['1v1_torneos_1'] = None
                st.session_state['1v1_step'] = 2
                st.rerun()
        else:
            st.success(f"Seleccionaste: **{st.session_state['1v1_jugador_1_display']}**")
            if st.button("🔄 Cambiar jugador"):
                reset_jugador()
                st.rerun()

# Paso 2: Selección de torneos para el jugador base
if st.session_state['1v1_step'] == 2 or st.session_state['1v1_torneos_1'] is None:
    expander_title_torneos = "2️⃣ Confirmar torneos"
else:
    expander_title_torneos = "2️⃣ Confirmar torneos"

if st.session_state['1v1_jugador_1_display']:
    df_jug_1 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_1_display']]
    if df_jug_1.empty:
        st.warning("El jugador seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugador()
        st.stop()
    jugador_1_row = df_jug_1.iloc[0]

    # Nombres seguros para UI
    j1_name = str(jugador_1_row.get('ID_Display', jugador_1_row.get('Nombre_transfermarket', 'Jugador base')))

    torneos_disp_1 = df_all[df_all["ID"] == jugador_1_row["ID"]]["Torneo"].dropna().tolist()
    torneos_disp_1 = sorted(set(t for sublist in torneos_disp_1 for t in (sublist if isinstance(sublist, list) else [sublist])))

    with col_izq:
        with st.expander(expander_title_torneos, expanded=True):
            if st.session_state['1v1_step'] == 2 or st.session_state['1v1_torneos_1'] is None:
                with st.form("seleccion_torneos"):
                    col1, col2 = st.columns([1,1])
                    with col1:
                        torneos_sel_1 = st.multiselect(
                            f"Torneos de {j1_name}",
                            torneos_disp_1,
                            default=torneos_disp_1,
                            key="1v1_torneos_1_select"
                        )
                    submit_torneos = st.form_submit_button("Confirmar torneos")
                if submit_torneos:
                    # Guardar torneos confirmados
                    st.session_state['1v1_torneos_1'] = torneos_sel_1

                    # --- Overrides para sidebar según Jugador base ---
                    try:
                        df_j1_scope = df_all[df_all['ID'].astype(str) == str(jugador_1_row['ID'])].copy()
                        # Limitar por torneos seleccionados
                        df_j1_scope = df_j1_scope[df_j1_scope['Torneo'].apply(lambda x: any(t in (x if isinstance(x, list) else [x]) for t in torneos_sel_1))]
                        # Temporada (tomamos la de la fila seleccionada si está)
                        st.session_state['1v1_sync_temporada'] = jugador_1_row.get('Temporada', None)
                        # País o países involucrados en esos torneos
                        if 'Pais' in df_j1_scope.columns:
                            paises_sync = sorted(df_j1_scope['Pais'].dropna().astype(str).unique().tolist())
                        else:
                            paises_sync = []
                        # Fallback a país del jugador si no se pudo inferir de torneos
                        if not paises_sync and 'Pais' in jugador_1_row.index and pd.notna(jugador_1_row['Pais']):
                            paises_sync = [str(jugador_1_row['Pais'])]
                        st.session_state['1v1_sync_paises'] = paises_sync
                        # Torneos seleccionados de Jugador base
                        st.session_state['1v1_sync_torneos'] = torneos_sel_1

                        # Posición general del Jugador base (derivada de su scope o, si no, del total en df_all)
                        pos_gen_sync = None
                        if 'Posicion_general' in df_j1_scope.columns:
                            vals = df_j1_scope['Posicion_general'].dropna().astype(str).unique().tolist()
                            if vals:
                                pos_gen_sync = vals[0]
                        if not pos_gen_sync and 'Posicion_general' in df_all.columns:
                            vals = df_all.loc[df_all['ID'].astype(str) == str(jugador_1_row['ID']), 'Posicion_general'] \
                                       .dropna().astype(str).unique().tolist()
                            if vals:
                                pos_gen_sync = vals[0]
                        st.session_state['1v1_sync_pos_gen'] = pos_gen_sync
                    except Exception:
                        pass

                    # Avanzar al step 3
                    st.session_state['1v1_step'] = 3
                    st.rerun()
            else:
                col_a, col_b = st.columns([1,1])
                with col_a:
                    st.success(
                        f"Torneos seleccionados para **{j1_name}**: {', '.join(st.session_state['1v1_torneos_1'])}"
                    )
                    if st.button("🔄 Cambiar torneos"):
                        reset_torneos()
                        st.rerun()
                # La ficha técnica completa se mostrará en la segunda columna en el Paso 3

# -------------------- BLOQUE 5: AGREGACIÓN Y FICHA TÉCNICA DEL JUGADOR --------------------
# Paso 3: Si todo confirmado, resto de la lógica
if st.session_state.get('1v1_step') == 3:
    # Obtener SIEMPRE el objeto correcto (Series) para jugador_1_row
    df_jug_1 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_1_display']]
    if df_jug_1.empty:
        st.warning("El jugador seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugador()
        st.stop()
    jugador_1_row = df_jug_1.iloc[0]
    torneos_1 = st.session_state['1v1_torneos_1']

    # -------------------- BLOQUE 5A: construir muestra final y agregar (solo jugador base) --------------------
    def extraer_muestra_jugador(df_base: pd.DataFrame, jugador_id, torneos: list) -> pd.DataFrame:
        df_b = df_base.copy()
        df_b["Torneo"] = df_b["Torneo"].apply(lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else []))
        jug_id_str = str(jugador_id)
        return df_b[(df_b["ID"].astype(str) == jug_id_str) & (df_b["Torneo"].apply(lambda ts: any(t in ts for t in torneos)))]

    # Partimos de la muestra filtrada (df) y añadimos los registros del jugador según torneos elegidos
    df_union = df.copy()
    df_j1_extra = extraer_muestra_jugador(df_all, jugador_1_row["ID"], torneos_1)
    if not df_j1_extra.empty:
        df_union = pd.concat([df_union, df_j1_extra], ignore_index=True)

    # Agregación por ID (suma numéricas excepto Edad, identidad por first, Torneo como lista única)
    if "Torneo" in df_union.columns:
        columnas_sumables = [c for c in df_union.columns if is_numeric_dtype(df_union[c]) and c not in ["Edad"]]
        agg_spec = {col: "sum" for col in columnas_sumables}
        opcionales_first = [
            "Nombre_transfermarket", "ID_Equipo", "logo_equipo", "Equipo_data",
            "Posicion_general", "Posicion_detallada", "Pais",
            "Pais_diminutivo", "Nacionalidad_2", "Nacionalidad",
            "Edad", "Temporada", "Color primario", "Equipo_data_full"
        ]
        for col in opcionales_first:
            if col in df_union.columns:
                agg_spec[col] = "first"
        agg_spec["Torneo"] = lambda x: list(set(t for lst in x.apply(lambda v: v if isinstance(v, list) else [v]) for t in lst if pd.notna(t)))
        df_muestra_agg = df_union.groupby("ID", as_index=False).agg(agg_spec)
    else:
        df_muestra_agg = df_union.copy()

    # Aplicar métricas personalizadas y preparar columnas finales
    df_muestra_proc, considerar_dict, tipos_dict, metricas_porcentaje, metricas_invertir = aplicar_metricas_personalizadas(df_muestra_agg, df_metricas)

    if all(col in df_muestra_proc.columns for col in ["Equipo_data", "Pais_diminutivo"]) and "Equipo_data_full" not in df_muestra_proc.columns:
        df_muestra_proc["Equipo_data_full"] = df_muestra_proc["Equipo_data"].astype(str) + " " + df_muestra_proc["Pais_diminutivo"].astype(str)

    # Ficha del jugador agregada (para siguiente etapa)
    j1_agg = df_muestra_proc.loc[df_muestra_proc["ID"].astype(str) == str(jugador_1_row["ID"])].copy()
    if j1_agg.empty:
        st.warning("No se pudo preparar la muestra agregada para el jugador. Revisa los torneos seleccionados.")
        st.stop()

    jugador_1 = j1_agg.iloc[0]

    # Dejar lista la base procesada para pasos siguientes
    df_agg = df_muestra_proc.copy()

    # -------------------- BLOQUE 5B: FICHA TÉCNICA DEL JUGADOR SELECCIONADO --------------------
    with col_der_placeholder:
        st.markdown("---")
        st.markdown("### Ficha técnica del jugador seleccionado")
        # -- Estilos modernos para tarjeta de jugador (ligero y responsivo)
        st.markdown(
            """
            <style>
            .player-card{display:flex;gap:16px;padding:14px 16px;border-radius:14px;
                          background:linear-gradient(180deg,rgba(0,0,0,.04),rgba(0,0,0,.02));
                          border:1px solid rgba(0,0,0,.08);box-shadow:0 2px 8px rgba(0,0,0,.06);position:relative}
            .player-card:before{content:'';position:absolute;inset:0;border-top:4px solid var(--accent,#444);border-radius:14px}
            .pc-left{display:flex;flex-direction:column;align-items:center;gap:8px;min-width:84px}
            .pc-avatar{width:68px;height:68px;border-radius:50%;object-fit:cover;filter:grayscale(15%)}
            .pc-crest{width:52px;height:52px;object-fit:contain;border-radius:10px;background:#fff;padding:4px;border:1px solid rgba(0,0,0,.08)}
            .pc-right{flex:1}
            .pc-name{font-weight:700;font-size:1.2rem;line-height:1.2}
            .pc-sub{opacity:.7;font-size:.85rem;margin-top:2px}
            .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75rem;margin-right:6px;background:rgba(0,0,0,.06)}
            .badge.accent{background:var(--accent,#444);color:#fff}
            .pc-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 14px;margin-top:10px}
            .stat .label{font-size:.78rem;opacity:.7}
            .stat .value{font-weight:600}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Función para mostrar la ficha técnica de un jugador (tarjeta moderna)
    def mostrar_ficha(jugador):
        # Utilidades locales
        def fmt_int(x):
            try:
                return f"{int(float(x)):,}".replace(",", ".")
            except Exception:
                return "N/D"
        def fmt_float(x, nd=1):
            try:
                v = float(x)
                return (f"{v:.{nd}f}").replace(".", ",")
            except Exception:
                return "N/D"

        # Campos seguros
        equipo_full = jugador.get("Equipo_data_full")
        if not equipo_full:
            equipo_full = (str(jugador.get("Equipo_data", "")).strip() + (" " + str(jugador.get("Pais_diminutivo", "")).strip() if pd.notnull(jugador.get("Pais_diminutivo", None)) else "")).strip() or "N/D"
        nombre = jugador.get("ID_Display") or jugador.get("Nombre_transfermarket") or "Jugador"
        nombre = _sanitize_text(nombre)
        temporada = _sanitize_text(jugador.get("Temporada", "N/D"))
        pos_gen = _sanitize_text(jugador.get("Posicion_general", "N/D"))
        pos_det = _sanitize_text(jugador.get("Posicion_detallada", "N/D"))
        edad = fmt_int(jugador.get("Edad", None))
        nac = _sanitize_text(jugador.get("Nacionalidad", jugador.get("Nacionalidad_2", "N/D")))
        pie = _sanitize_text(jugador.get("Pie", "N/D"))
        min_jug = fmt_int(jugador.get("Minutos_jugados", None))
        m90s = fmt_float(jugador.get("M90s_jugados", None), 1)

        # Valor mercado
        valor_mercado = jugador.get("Valor_mercado", None)
        if isinstance(valor_mercado, (int, float, np.number)) and pd.notnull(valor_mercado):
            valor_mercado_str = f"€{valor_mercado/1_000_000:.1f}M" if valor_mercado >= 1_000_000 else f"€{valor_mercado/1_000:.0f}k"
        else:
            valor_mercado_str = "N/D"

        # Logos y colores
        logo = jugador.get("logo_equipo", "")
        avatar_url = "https://img.icons8.com/ios-filled/100/000000/user.png"
        accent = _clean_hex_color(jugador.get("Color primario", "")) or "#5B8FF9"

        crest_html = f"<img class='pc-crest' src='{_sanitize_text(logo)}'/>" if isinstance(logo, str) and logo.strip() else ""

        html = f"""
        <div class='player-card' style='--accent:{accent}'>
          <div class='pc-left'>
            <img class='pc-avatar' src='{avatar_url}'/>
            {crest_html}
          </div>
          <div class='pc-right'>
            <div class='pc-name'>{_sanitize_text(nombre)}</div>
            <div class='pc-badges'>
              <span class='badge accent'>{_sanitize_text(pos_gen)}</span>
              <span class='badge'>{_sanitize_text(pos_det)}</span>
              <span class='badge'>{_sanitize_text(edad)} años</span>
            </div>
            <div class='pc-sub'>{_sanitize_text(equipo_full)} • {temporada}</div>
            <div class='pc-stats'>
              <div class='stat'><div class='label'>Minutos jugados</div><div class='value'>{min_jug}</div></div>
              <div class='stat'><div class='label'>Partidos completos (M90s)</div><div class='value'>{m90s}</div></div>
              <div class='stat'><div class='label'>Nacionalidad</div><div class='value'>{nac}</div></div>
              <div class='stat'><div class='label'>Pie</div><div class='value'>{pie}</div></div>
              <div class='stat'><div class='label'>Valor mercado</div><div class='value'>{valor_mercado_str}</div></div>
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # Mostrar la ficha en la **segunda** columna
    with col_der_placeholder:
        mostrar_ficha(jugador_1)

    # -------------------- BLOQUE 6: MUESTRA DE COMPARACIÓN --------------------
    # Usar SIEMPRE la muestra filtrada del sidebar
    df_muestra = df.copy()

    # Excluir jugador seleccionado de la muestra para percentiles comparativos
    df_muestra = df_muestra[~df_muestra["ID"].isin([jugador_1["ID"]])]

    # ---- Texto estilo página 1 ----
    n_jugadores = df_muestra["ID"].nunique() if "ID" in df_muestra.columns else len(df_muestra)

    # Temporada
    temporada_txt = (
        ", ".join(sorted(df_muestra["Temporada"].dropna().astype(str).unique()))
        if "Temporada" in df_muestra.columns else str(temporada)
    )

    # Países (<=2 lista; si no, "Múltiples países")
    if "Pais" in df_muestra.columns:
        paises_uni = sorted(df_muestra["Pais"].dropna().astype(str).unique())
        paises_txt = ", ".join(paises_uni) if len(paises_uni) <= 2 else "Múltiples países"
    else:
        paises_txt = ", ".join(paises_sel) if len(paises_sel) <= 2 else "Múltiples países"

    # Torneos (<=4 lista; si no, "Múltiples torneos")
    if "Torneo" in df_muestra.columns:
        tor_set = []
        for val in df_muestra["Torneo"].dropna():
            if isinstance(val, list):
                tor_set.extend([str(t).strip() for t in val])
            else:
                tor_set.append(str(val).strip())
        tor_uni = sorted(set(tor_set))
        torneos_txt = ", ".join(tor_uni) if len(tor_uni) <= 4 else "Múltiples torneos"
    else:
        torneos_txt = ", ".join(torneos_sel) if len(torneos_sel) <= 4 else "Múltiples torneos"

    # Nacionalidad (respetando el toggle de detallada)
    if usar_nacionalidad_detallada and "Nacionalidad" in df_muestra.columns:
        _nacs = sorted(df_muestra["Nacionalidad"].dropna().astype(str).unique())
    elif "Nacionalidad_2" in df_muestra.columns:
        _nacs = sorted(df_muestra["Nacionalidad_2"].dropna().astype(str).unique())
    else:
        _nacs = []
    nacionalidad_txt = ", ".join(_nacs) if len(_nacs) <= 3 else "Múltiples nacionalidades"

    # Posición general / detallada
    pos_gen_txt = (
        ", ".join(sorted(set(pos_gen_sel))) if pos_gen_sel else "Todas"
    )
    pos_det_txt = (
        "Todas" if set(pos_det_sel) == set(pos_det_opciones) else ", ".join(sorted(set(pos_det_sel)))
    )

    # Info estilo página 1 (4 líneas)
    info_md = (
        f"La muestra filtrada contiene **{n_jugadores} jugadores únicos**.  \n"
        f"**Temporada:** {temporada_txt} - **País del equipo:** {paises_txt} - **Torneo:** {torneos_txt}  \n"
        f"**Minutos jugados:** >{min_jugados} - **Partidos completos (M90s):** >{min_m90s} - **Edad:** {edad_range[0]}-{edad_range[1]} - **Nacionalidad:** {nacionalidad_txt}.  \n"
        f"**Posición general:** {pos_gen_txt} - **Posición detallada:** {pos_det_txt}"
    )
    st.info(info_md)

    # Guardar una descripción simple por si otros bloques la requieren (radar)
    st.session_state["muestra_percentil"] = "Muestra del sidebar"

    # -------------------- BLOQUE 7: SELECCIÓN DE MÉTRICAS PARA RADAR (WIZARD AVANZADO) --------------------
    st.markdown("---")
    st.markdown("### Comparativa del jugador en base a percentiles")

    # --- Selector avanzado de métricas por tipo relevante ---
    pos_det_j1 = jugador_1.get("Posicion_detallada", "")
    pos_gen_j1 = jugador_1.get("Posicion_general", "")
    nombre_j1 = jugador_1.get("Nombre_transfermarket", "")

    # Armar bloques de métricas por tipo relevante
    bloques_metricas_wizard = {
        "Métricas Físicas": metricas_fisicas,
        "Métricas de Construcción": metricas_construccion_general + metricas_construccion_ofensiva + metricas_centros,
        "Métricas Ofensivas": metricas_ofensivas + metricas_balon_parado,
    }
    diminutivo_pos_det = diminutivos_pos.get(pos_det_j1, pos_det_j1)
    if diminutivo_pos_det == "PT":
        bloques_metricas_wizard = {
            "Métricas de Portero": metricas_portero,
            "Métricas Físicas": metricas_fisicas,
            "Métricas de Construcción": metricas_construccion_general,
        }

    # Determinar modo (por 90 o totales)
    if "wizard_metricas_1v1" not in st.session_state:
        st.session_state["wizard_metricas_1v1"] = {
            "modo_90": True,
            "metricas_por_tipo": {},
        }
    modo_90 = st.toggle("Totales ↔ Por 90", value=st.session_state["wizard_metricas_1v1"].get("modo_90", True), key="modo_90_1v1")

    # Filtrar métricas disponibles por modo
    tipos_validos = ["/90", "Porcentaje"] if modo_90 else ["Totales", "Porcentaje"]
    metricas_validas = [m for m, tipo in considerar_dict.items() if tipo in tipos_validos]

    # Sugerencias por posición para cada bloque
    metricas_default_posicion = metricas_default_por_posicion.get(pos_det_j1, [])
    # Asegurar que TODAS las métricas default por posición estén representadas en algún bloque
    default_all = [m for m in metricas_default_posicion if m in considerar_dict]
    # Unión de métricas incluidas en los bloques actuales
    metricas_en_bloques = set()
    for _b_mets in bloques_metricas_wizard.values():
        metricas_en_bloques.update(_b_mets)
    faltantes_defaults = [m for m in default_all if m not in metricas_en_bloques]
    if faltantes_defaults:
        # Bloque catch-all para defaults no cubiertos por los bloques temáticos
        bloques_metricas_wizard["Otras métricas (sugeridas)"] = faltantes_defaults
    # Preselección por bloque
    metricas_por_tipo_default = {}
    for bloque, metricas_bloque in bloques_metricas_wizard.items():
        metricas_bloque_validas = [m for m in metricas_bloque if m in metricas_validas]
        sugeridas = [m for m in metricas_default_posicion if m in metricas_bloque_validas]
        metricas_por_tipo_default[bloque] = sugeridas

    # --- Resetear selección de métricas si cambia el jugador o la posición detallada ---
    id_jugador_actual = jugador_1["ID"]
    pos_det_actual = jugador_1["Posicion_detallada"]
    if (
        "wizard_metricas_1v1_last_id" not in st.session_state
        or st.session_state["wizard_metricas_1v1_last_id"] != id_jugador_actual
        or st.session_state.get("wizard_metricas_1v1_last_posdet") != pos_det_actual
    ):
        st.session_state["wizard_metricas_1v1"]["metricas_por_tipo"] = dict(metricas_por_tipo_default)
        st.session_state["wizard_metricas_1v1_last_id"] = id_jugador_actual
        st.session_state["wizard_metricas_1v1_last_posdet"] = pos_det_actual

    # Usar selección previa si existe, sino sugerida
    metricas_por_tipo_sel = st.session_state["wizard_metricas_1v1"].get("metricas_por_tipo", metricas_por_tipo_default)

    st.info(f"ℹ️ Se han preseleccionado métricas sugeridas para la posición de {nombre_j1}: {pos_det_j1}. Si deseas puedes añadir o modificar las métricas para la comparativa. Puedes seleccionar entre 5 y 12 métricas, agrupadas por tipo relevante.")
    with st.expander("Modificar métricas a analizar", expanded=False):
        seleccionadas_por_tipo = {}
        total_sel = 0
        for bloque, metricas_bloque in bloques_metricas_wizard.items():
            metricas_bloque_validas = [m for m in metricas_bloque if m in metricas_validas]
            # --- PATCH: Sugeridas solo válidas respecto a opciones actuales ---
            sugeridas_previas = metricas_por_tipo_sel.get(bloque, [])
            sugeridas = [m for m in sugeridas_previas if m in metricas_bloque_validas]
            if not sugeridas:
                sugeridas = [m for m in metricas_default_posicion if m in metricas_bloque_validas][:3]
            seleccionadas = st.multiselect(
                f"{bloque}",
                options=metricas_bloque_validas,
                default=sugeridas,
                key=f"wizard_{bloque}_1v1"
            )
            seleccionadas_por_tipo[bloque] = seleccionadas
            total_sel += len(seleccionadas)
        st.markdown(f"**Total de métricas seleccionadas:** {total_sel}")
        if total_sel < 5 or total_sel > 12:
            st.warning("⚠️ Debes seleccionar entre 5 y 12 métricas en total (sumando todos los bloques).")
        aplicar = st.button("Aplicar cambios", key="wizard_aplicar_1v1")
        if aplicar:
            if 5 <= total_sel <= 12:
                todas_metricas = []
                tipos_sel = []
                for bloque, mets in seleccionadas_por_tipo.items():
                    todas_metricas.extend(mets)
                    if mets:
                        tipos_sel.append(bloque)
                st.session_state["1v1_metricas_sel"] = todas_metricas
                st.session_state["1v1_tipos_sel"] = tipos_sel
                st.session_state["wizard_metricas_1v1"]["modo_90"] = modo_90
                st.session_state["wizard_metricas_1v1"]["metricas_por_tipo"] = seleccionadas_por_tipo
                st.success("Selección de métricas actualizada. Se usará en los gráficos y tablas.")
                st.rerun()
            else:
                st.warning("Debes seleccionar entre 5 y 12 métricas en total.")

    # Usar selección válida (con fallback a sugerencia)
    seleccionadas = []
    if "1v1_metricas_sel" in st.session_state:
        seleccionadas = [m for m in st.session_state["1v1_metricas_sel"] if m in metricas_validas]
    if not seleccionadas:
        # fallback: juntar sugeridas de todos los bloques y limitar a 12
        seleccionadas = []
        for bloque in bloques_metricas_wizard:
            seleccionadas += metricas_por_tipo_default.get(bloque, [])
        if len(seleccionadas) > 12:
            seleccionadas = seleccionadas[:12]
    if len(seleccionadas) < 5:
        st.warning("⚠️ Debes seleccionar al menos 5 métricas.")
        st.stop()
    if len(seleccionadas) > 12:
        seleccionadas = seleccionadas[:12]

    # Construir DataFrame de comparación (muestra del sidebar procesada con métricas)
    df_comparacion, _, _, _, metricas_invertir_ctx = aplicar_metricas_personalizadas(df_muestra.copy(), df_metricas)

    # Definir el título y subtítulo del gráfico de radar
    titulo_grafico = (
        f"{jugador_1['Nombre_transfermarket']} {jugador_1['Temporada']} ({jugador_1['Equipo_data']})"
    )

    # Posiciones de la muestra para el subtítulo (si existen)
    posiciones_muestra = (
        df_comparacion.get("Posicion_detallada", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    pos_det_dimin = list(dict.fromkeys([diminutivos_pos.get(p, p) for p in posiciones_muestra]))

    # Torneos de contexto: usamos los seleccionados en el sidebar
    torneos = torneos_sel

    # Temporada de contexto (si hay varias, mostramos lista ordenada)
    _temps = (
        df_comparacion.get("Temporada", pd.Series([temporada]))
        .dropna().astype(str).unique().tolist()
    )
    temporada_txt_ctx = ", ".join(sorted(_temps)) if _temps else str(temporada)

    subtitulo_grafico = (
        f"Comparativa vs {', '.join(pos_det_dimin) if pos_det_dimin else 'muestra filtrada'} "
        f"de {', '.join(torneos)} en {temporada_txt_ctx}"
    )

    # Limitar el número de métricas seleccionadas para los gráficos radar
    max_metricas = 12
    if len(seleccionadas) > max_metricas:
        seleccionadas = seleccionadas[:max_metricas]

    # Ordenar métricas seleccionadas por bloque y luego alfabéticamente
    bloques_orden = [
        metricas_portero, metricas_fisicas, metricas_centros,
        metricas_construccion_general, metricas_construccion_ofensiva,
        metricas_ofensivas, metricas_balon_parado
    ]
    seleccionadas_radar = []
    for bloque in bloques_orden:
        seleccionadas_en_bloque = sorted([m for m in seleccionadas if m in bloque])
        seleccionadas_radar.extend(seleccionadas_en_bloque)

    st.info(f"🔢 Máximo de métricas permitidas : {max_metricas}")

    # Forzar el modo según el tema actual
modo_claro = (get_theme_type() == "light")
# ==================== BLOQUE FINAL: VISUALIZACIÓN pizza ====================
st.markdown("### Comparativas visuales del jugador")
st.markdown("#### Pizza principal por métricas seleccionadas")

# Determinar lista final de métricas para el gráfico (ordenada por bloques)
try:
    metricas_finales = list(seleccionadas_radar) if len(seleccionadas_radar) > 0 else list(seleccionadas)
except Exception:
    metricas_finales = list(seleccionadas)

# Contexto: usar la muestra ya transformada con métricas (df_comparacion)
df_viz = df_comparacion[df_comparacion[metricas_finales].notna().all(axis=1)].copy()
if df_viz.empty:
    st.warning("⚠️ No hay suficientes jugadores para calcular percentiles en la muestra actual.")
    st.stop()

# Encabezados
titulo_grafico = f"{jugador_1['Nombre_transfermarket']} {jugador_1['Temporada']} ({jugador_1['Equipo_data']})"
posiciones_dimin = list(dict.fromkeys([
    diminutivos_pos.get(p, p)
    for p in df_comparacion.get("Posicion_detallada", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
]))
_torneos_base = st.session_state.get('1v1_torneos_1', torneos_sel)
torneos_str = ', '.join([t for t in _torneos_base if t])
_temporada_j1 = jugador_1.get("Temporada", temporada)
subtitulo_grafico = f"vs {', '.join(posiciones_dimin) if posiciones_dimin else 'muestra filtrada'} de {torneos_str} en {_temporada_j1}"

col_izq_p, col_centro_p, col_der_p = st.columns([0.3, 0.4, 0.3])
with col_centro_p:
    fig = crear_pizza_chart(
        jugador=jugador_1,
        df_contexto=df_viz,
        metricas=metricas_finales,
        considerar_dict=considerar_dict,
        tipos_dict=tipos_dict,
        metricas_invertir=metricas_invertir_ctx,
        titulo=titulo_grafico,
        subtitulo=None,
        modo_claro=modo_claro,
    )
    st.pyplot(fig)

#
# Caption centrado bajo la gráfica principal (antes de la tabla)

def _center_caption(md_text: str) -> str:
    html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md_text)
    html = html.replace("\n", "<br>")
    return f"<div style='text-align:center; font-size:0.9rem; opacity:0.85'>{html}</div>"

st.markdown(_center_caption(info_md), unsafe_allow_html=True)

# Tabla moderna: valores y percentiles por métrica (mono‑jugador)
st.markdown("#### Valores y percentiles por métrica")

# Calcular percentiles por métrica vs muestra (df_comparacion no incluye al jugador)
filas = []
for m in metricas_finales:
    # Valor del jugador
    v = jugador_1.get(m, np.nan)
    # Serie de la muestra
    serie = pd.to_numeric(df_comparacion[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(serie) == 0 or pd.isna(v):
        pct = np.nan
    else:
        try:
            v_f = float(v)
        except Exception:
            v_f = np.nan
        if pd.isna(v_f):
            pct = np.nan
        else:
            # Percentil: proporción de valores <= v
            pct = float((serie <= v_f).mean()) * 100.0
            # Si la métrica se invierte (menor es mejor), reflejar percentil
            if isinstance(metricas_invertir_ctx, (set, list, dict)) and (m in metricas_invertir_ctx):
                pct = 100.0 - pct
    filas.append({
        "Métrica": m,
        "Valor": formatear_valor(m, v),
        "Percentil": None if pd.isna(pct) else int(round(np.clip(pct, 0, 100)))
    })

# Renderizar en HTML con estilo moderno y barra de porcentaje por percentil
tbl_css = """
<style>
.table-modern{width:100%;border-collapse:separate;border-spacing:0 8px}
.table-modern th{font-size:.85rem;text-align:left;opacity:.7;padding:6px 10px}
.table-modern td{background:rgba(0,0,0,.03);padding:10px;border:1px solid rgba(0,0,0,.06)}
.table-modern td:first-child{border-top-left-radius:10px;border-bottom-left-radius:10px}
.table-modern td:last-child{border-top-right-radius:10px;border-bottom-right-radius:10px}
.badge-metric{font-weight:600}
.pbar-wrap{display:flex;align-items:center;gap:10px}
.pbar{flex:1;height:10px;border-radius:999px;background:rgba(0,0,0,.08);overflow:hidden}
.pbar>span{display:block;height:100%;border-radius:inherit}
.pct-label{width:42px;text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
</style>
"""

rows_html = []
for r in filas:
    pct = r["Percentil"]
    # Color en HSL del rojo (0) al verde (120)
    hue = 120 if pct is None else int(round((pct/100.0)*120))
    width = 0 if pct is None else pct
    bar_html = f"<div class='pbar'><span style='width:{width}%;background:hsl({hue},70%,45%);'></span></div>"
    pct_html = "N/D" if pct is None else f"{pct}%"
    rows_html.append(
        f"<tr>"
        f"<td class='badge-metric'>{_sanitize_text(r['Métrica'])}</td>"
        f"<td>{_sanitize_text(r['Valor'])}</td>"
        f"<td><div class='pbar-wrap'>{bar_html}<div class='pct-label'>{pct_html}</div></div></td>"
        f"</tr>"
    )

tabla_html = (
    tbl_css +
    "<table class='table-modern'>"
    "<thead><tr><th>Métrica</th><th>Valor</th><th>Percentil</th></tr></thead>"
    "<tbody>" + "".join(rows_html) + "</tbody></table>"
)
st.markdown(tabla_html, unsafe_allow_html=True)

st.markdown("#### Comparativas por tipo de métrica")
bloques = [
    ("Físicas", metricas_fisicas, "#6FBF73"),
    ("Centros", metricas_centros, "#2F80ED"),
    ("Construcción general", metricas_construccion_general, "#2F80ED"),
    ("Construcción ofensiva", metricas_construccion_ofensiva, "#2F80ED"),
    ("Ofensivas", metricas_ofensivas, "#EB5757"),
    ("Balón parado", metricas_balon_parado, "#EB5757")
]
for i in range(0, len(bloques), 3):
    col1, col2, col3 = st.columns(3)
    for (titulo, lista_metricas, color), col in zip(bloques[i:i+3], [col1, col2, col3]):
        considerar_tipo = ["/90", "Porcentaje"] if modo_90 else ["Totales", "Porcentaje"]
        metricas_filtradas = [
            m for m in lista_metricas
            if considerar_dict.get(m) in considerar_tipo and m in df_comparacion.columns
        ]
        if not metricas_filtradas:
            continue
        fig = crear_pizza_chart(
            jugador=jugador_1,
            df_contexto=df_comparacion,
            metricas=metricas_filtradas,
            considerar_dict=considerar_dict,
            tipos_dict=tipos_dict,
            metricas_invertir=metricas_invertir_ctx,
            titulo=f"{titulo} – {jugador_1['Nombre_transfermarket']} {jugador_1['Temporada']} ({jugador_1['Equipo_data']})",
            subtitulo=None,
            modo_claro=modo_claro,
        )
        with col:
            st.pyplot(fig)