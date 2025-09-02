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



from utils.utils_visuals import (
    crear_radar_percentil_plotly
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
# Mostrar temporadas como enteros y filtrar de forma robusta (2025, 2025.0, "2025.0" -> 2025)
if "Temporada" in df.columns:
    _temp_series_num = pd.to_numeric(df["Temporada"], errors="coerce").dropna()
    temporadas_disponibles = sorted(
        _temp_series_num.round().astype(int).unique().tolist(), reverse=True
    )
else:
    temporadas_disponibles = []

if not temporadas_disponibles:
    st.error("No hay temporadas disponibles en los datos.")
    st.stop()

# Default a override (si existe) → 2025 → más reciente
override_temp = st.session_state.get('1v1_sync_temporada')
try:
    override_int = int(round(float(str(override_temp)))) if override_temp is not None else None
except Exception:
    override_int = None

if override_int in temporadas_disponibles:
    default_temporada = override_int
elif 2025 in temporadas_disponibles:
    default_temporada = 2025
else:
    default_temporada = temporadas_disponibles[0]

idx_temp = temporadas_disponibles.index(default_temporada) if default_temporada in temporadas_disponibles else 0

temporada = st.sidebar.selectbox(
    "Temporada",
    temporadas_disponibles,
    index=idx_temp
)

# Igualamos por entero con conversión robusta del DF
if "Temporada" in df.columns:
    _temp_col = pd.to_numeric(df["Temporada"], errors="coerce").round().astype("Int64")
    df_temp = df[_temp_col == int(temporada)].copy()
else:
    df_temp = df.copy()

# --- País del equipo ---
paises_opciones = sorted(df_temp["Pais"].dropna().unique()) if "Pais" in df_temp.columns else []
default_pais = "Peru" if "Peru" in paises_opciones else (paises_opciones[0] if paises_opciones else None)
# --- PATCH: override paises desde session_state (sanitizado) ---
override_paises = st.session_state.get('1v1_sync_paises')
if override_paises is not None and not isinstance(override_paises, (list, tuple, set)):
    override_paises = [override_paises]
# Quedarnos solo con los que existen en opciones
safe_paises_default = [p for p in (override_paises or []) if p in paises_opciones]
if not safe_paises_default:
    safe_paises_default = [default_pais] if (default_pais and default_pais in paises_opciones) else paises_opciones

paises_sel = st.sidebar.multiselect(
    'País del equipo',
    paises_opciones,
    default=safe_paises_default
)

# --- Torneo ---
# Opciones de torneo según países seleccionados en el contexto actual
if paises_sel:
    torneos_opciones = sorted(
        df_temp.loc[df_temp["Pais"].isin(paises_sel), "Torneo"]
        .dropna().astype(str).str.strip().unique()
    )
else:
    torneos_opciones = sorted(
        df_temp["Torneo"].dropna().astype(str).str.strip().unique()
    )

# Fallback por si no hubiese torneos para los países elegidos:
if not torneos_opciones:
    torneos_opciones = sorted(
        df_temp["Torneo"].dropna().astype(str).str.strip().unique()
    )

# Clave de contexto (Temporada + Países) para resetear torneos automáticamente
ctx_tor_hash = f"{int(temporada)}|{'/'.join(sorted(map(str, paises_sel)))}"
if st.session_state.get("radar_ctx_tor_hash") != ctx_tor_hash:
    # Al cambiar el contexto, preseleccionamos TODOS los torneos válidos
    st.session_state["torneos_sel_radar"] = torneos_opciones[:]
    st.session_state["radar_ctx_tor_hash"] = ctx_tor_hash

# Sanitizar selección previa con respecto a las opciones actuales
sel_actual = [
    t for t in st.session_state.get("torneos_sel_radar", []) if t in torneos_opciones
] or torneos_opciones[:]
st.session_state["torneos_sel_radar"] = sel_actual

# Crear el widget (sin `default=` para evitar el warning de Streamlit)
torneos_sel = st.sidebar.multiselect(
    "Torneo",
    torneos_opciones,
    key="torneos_sel_radar",
)

# Si el usuario deja vacío el multiselect, lo reponemos con todas las opciones
if len(torneos_sel) == 0 and len(torneos_opciones) > 0:
    st.session_state["torneos_sel_radar"] = torneos_opciones[:]
    st.rerun()

# DF filtrado preliminar (pais/torneo) – hacemos copy para evitar SettingWithCopy
mask_base = df_temp["Pais"].isin(paises_sel) & df_temp["Torneo"].isin(torneos_sel)
df_filtros = df_temp[mask_base].copy()

# Auto‑fix: si la combinación actual deja la muestra vacía, cargamos todos los
# torneos válidos para los países seleccionados (o, si tampoco hubiese, todos).
if df_filtros.empty:
    torneos_fallback = sorted(
        df_temp.loc[df_temp["Pais"].isin(paises_sel), "Torneo"]
        .dropna().astype(str).str.strip().unique()
    )
    if torneos_fallback and set(torneos_sel).isdisjoint(torneos_fallback):
        st.session_state["torneos_sel_radar"] = torneos_fallback
        st.rerun()
    # Fallback global (p. ej. si no hay datos para los países elegidos)
    torneos_all = sorted(df_temp["Torneo"].dropna().astype(str).str.strip().unique())
    if torneos_all and set(torneos_sel).isdisjoint(torneos_all):
        st.session_state["torneos_sel_radar"] = torneos_all
        st.rerun()


# -------------------- FILTROS RESTANTES EN SIDEBAR --------------------
with st.sidebar:

    # Minutos / M90s / Edad (defaults reactivos = 1/3 del máximo)
    # Contexto para reseteo automático cuando cambian Temporada/Torneo/País
    ctx_hash = f"{int(temporada)}|{';'.join(sorted(map(str, torneos_sel)))}|{';'.join(sorted(map(str, paises_sel)))}"

    # Máximos actuales en la muestra base (AGREGADOS POR ID)
    agg_max = (
        df_filtros.groupby("ID", as_index=False)[["Minutos_jugados", "M90s_jugados"]]
        .sum()
    ) if not df_filtros.empty else pd.DataFrame(columns=["Minutos_jugados", "M90s_jugados"])
    min_jugados_max = int(agg_max["Minutos_jugados"].max()) if not agg_max.empty else 0
    max_m90         = int(agg_max["M90s_jugados"].max())     if not agg_max.empty else 0

    # Paso dinámico para minutos (mejor fidelidad del 1/3 según el tamaño de la muestra)
    if min_jugados_max >= 300:
        step_minjug = 50
    elif min_jugados_max >= 120:
        step_minjug = 20
    else:
        step_minjug = 10

    def _third_of(n, step=1, min_nonzero=True):
        """Devuelve ~1/3 de n ajustado al múltiplo de `step` más cercano.
        Si queda 0 y hay muestra, opcionalmente asegura al menos `step`."""
        try:
            n = int(n)
        except Exception:
            return 0
        if n <= 0:
            return 0
        raw = n / 3.0
        # Redondeo al múltiplo más cercano del paso
        val = int(round(raw / step)) * step
        if min_nonzero and val == 0 and n > 0:
            val = step
        return min(val, n)

    # —— NUEVO CRITERIO DE DEFAULT ——
    # En lugar de 1/3 del máximo global, usamos:
    #   1/3 del “máximo de minutos por país”, pero tomando el país cuyo máximo es más bajo.
    # Esto evita castigar a torneos/countries con calendarios más cortos.
    base_for_third = min_jugados_max
    try:
        if {"Pais", "ID", "Minutos_jugados"}.issubset(df_temp.columns):
            scope = df_temp.copy()
            # Limitar a países y torneos actualmente seleccionados
            if paises_sel:
                scope = scope[scope["Pais"].isin(paises_sel)]
            if torneos_sel:
                scope = scope[scope["Torneo"].isin(torneos_sel)]
            # Suma por jugador dentro de cada país
            mins_per_id = scope.groupby(["Pais", "ID"])["Minutos_jugados"].sum()
            per_pais_max = mins_per_id.groupby("Pais").max()
            if not per_pais_max.empty:
                # Elegimos el país cuyo "máximo de minutos" sea el más bajo
                base_for_third = int(per_pais_max.min())
    except Exception:
        pass

    default_min_jug = _third_of(base_for_third, step=step_minjug)

    # M90s por defecto siempre 0 (pocos lo usarán)
    default_min_m90 = 0

    # Reseteo cuando cambia el contexto
    if st.session_state.get("radar_ctx_hash_minmax") != ctx_hash:
        st.session_state["radar_min_jugados_value"] = default_min_jug
        st.session_state["radar_min_m90s_value"] = default_min_m90
        st.session_state["radar_ctx_hash_minmax"] = ctx_hash

    # Preinicializar para evitar warning (no mezclar value= con key)
    st.session_state.setdefault("radar_min_jugados_value", default_min_jug)
    st.session_state.setdefault("radar_min_m90s_value", default_min_m90)

    min_jugados = st.slider(
        "Minutos jugados (mínimo)",
        0,
        max(min_jugados_max, 0),
        step=step_minjug,
        key="radar_min_jugados_value",
    )

    min_m90s = st.slider(
        "Partidos completos jugados (M90s)",
        0,
        max(max_m90, 0),
        step=1,
        key="radar_min_m90s_value",
    )

    if "Edad" in df_filtros.columns:
        edad_min = int(df_filtros["Edad"].min()); edad_max = int(df_filtros["Edad"].max())
        default_edad_min = max(15, edad_min)
        edad_range = st.slider("Edad", edad_min, edad_max, (default_edad_min, edad_max))
    else:
        edad_range = (15, 100)

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

    # Determinar selección inicial sin tocar session_state (evita conflicto de Streamlit)
    desired_vis = default_vis
    if override_pos_gen and override_pos_gen in abbr_map:
        desired_vis = [abbr_map[override_pos_gen]]

    _seg = getattr(st, "segmented_control", None)
    if callable(_seg):
        try:
            sel_vis = _seg(
                "Posición general",
                opciones_vis,
                selection_mode="multi",
                default=desired_vis,
                help="Filtra por línea: POR, DEF, MED, DEL (en ese orden)."
            )
        except Exception:
            sel_vis = _seg(
                "Posición general",
                opciones_vis,
                default=desired_vis,
                help="Filtra por línea: POR, DEF, MED, DEL (en ese orden)."
            )
    else:
        sel_vis = st.multiselect(
            "Posición general",
            opciones_vis,
            default=desired_vis,
            help="Filtra por línea: POR, DEF, MED, DEL (en ese orden)."
        )

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


# --- Filtro final aplicado sobre df_temp (AGREGADO POR ID) ---
if df_temp.empty:
    st.warning("No hay datos para la temporada seleccionada.")
    st.stop()

# 1) Muestra base por País/Torneo (mantiene filas por torneo)
pre = df_temp[
    (df_temp["Pais"].isin(paises_sel)) &
    (df_temp["Torneo"].isin(torneos_sel))
].copy()

if pre.empty:
    st.warning("No hay datos para País/Torneo seleccionados.")
    st.stop()

# 2) Agregado por ID para evaluar condiciones de minutos/m90s/edad/posición/nacionalidad
agg = pre.groupby("ID").agg({
    "Minutos_jugados": "sum",
    "M90s_jugados": "sum",
    "Edad": "first",
    "Posicion_general": "first",
    "Posicion_detallada": (lambda s: s.dropna().iloc[0] if not s.dropna().empty else None),
    "Nacionalidad_2": "first",
    "Nacionalidad": "first",
}).reset_index()

# 3) Nacionalidad a usar según toggle
if usar_nacionalidad_detallada and "Nacionalidad" in agg.columns:
    nac_col = "Nacionalidad"
else:
    nac_col = "Nacionalidad_2" if "Nacionalidad_2" in agg.columns else None

# 4) Condiciones por ID
cond = (
    (agg["Minutos_jugados"] >= min_jugados) &
    (agg["M90s_jugados"]  >= min_m90s) &
    (agg["Edad"].between(edad_range[0], edad_range[1], inclusive="both")) &
    (agg["Posicion_general"].isin(pos_gen_sel))
)
if pos_det_sel:
    cond &= agg["Posicion_detallada"].isin(pos_det_sel)
if nac_col and len(nac2_sel) > 0:
    cond &= agg[nac_col].isin(nac2_sel)

ids_ok = agg.loc[cond, "ID"].astype(pre["ID"].dtype).tolist()

# 5) Muestra final: conservar TODAS las filas (torneos) de los IDs válidos
df = pre[pre["ID"].isin(ids_ok)].copy()

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


# --- Normalizar temporada en las etiquetas de selección (evitar 2025.0) ---
ids_disponibles = df_all.dropna(subset=["ID", "ID_Display"]).copy()
ids_disponibles = ids_disponibles[["ID", "ID_Display", "Equipo_data_full", "Temporada"]].drop_duplicates()

ids_disponibles["ID_Display"] = (
    ids_disponibles["ID_Display"].astype(str)
    .str.replace(r"(?<!\d)(\d{4})\.0(?!\d)", r"\1", regex=True)
)

# Etiqueta "limpia" para mostrar al usuario:
# - Elimina SOLO un prefijo literal ". " al inicio (no elimina iniciales como "K. ").
def _clean_id_display_label(s: str) -> str:
    s = str(s)
    s = re.sub(r"^\.\s+", "", s)  # quita ". " inicial
    return s.strip()

ids_disponibles["ID_Display_clean"] = ids_disponibles["ID_Display"].map(_clean_id_display_label)

# Ordenar por la etiqueta limpia (A→Z, ignorando tildes)
ids_disponibles = ids_disponibles.sort_values("ID_Display_clean", key=lambda s: s.map(_sort_key_az))

# Mapa: etiqueta limpia → etiqueta original (para guardar internamente el ID_Display real)
label_to_orig = dict(zip(ids_disponibles["ID_Display_clean"], ids_disponibles["ID_Display"]))

# Vista entera de Temprorada (por si se usa en otros textos de UI)
if "Temporada" in ids_disponibles.columns:
    ids_disponibles["Temporada_int"] = (
        pd.to_numeric(ids_disponibles["Temporada"], errors="coerce")
        .round()
        .astype("Int64")
    )

## -------------------- FLUJO DE SELECCIÓN DE JUGADORES Y TORNEOS --------------------
# Funciones auxiliares para resetear selección
def reset_jugadores():
    st.session_state['1v1_step'] = 1
    st.session_state['1v1_jugador_1_display'] = None
    st.session_state['1v1_jugador_2_display'] = None
    st.session_state['1v1_torneos_1'] = None
    st.session_state['1v1_torneos_2'] = None

def reset_torneos():
    st.session_state['1v1_step'] = 2
    st.session_state['1v1_torneos_1'] = None
    st.session_state['1v1_torneos_2'] = None

# Paso 1: Selección de jugadores con expander y título dinámico
expander_title = "1️⃣ Confirmar jugadores"
with st.expander(expander_title, expanded=True):
    if st.session_state['1v1_step'] == 1 or st.session_state['1v1_jugador_1_display'] is None or st.session_state['1v1_jugador_2_display'] is None:
        with st.form("seleccion_jugadores"):
            col1, col2 = st.columns(2)
            with col1:
                jugador_1_label = st.selectbox(
                    "Jugador 1",
                    ids_disponibles["ID_Display_clean"].tolist(),
                    key="1v1_jugador_1_select"
                )
            with col2:
                jugador_2_label = st.selectbox(
                    "Jugador 2",
                    ids_disponibles["ID_Display_clean"].tolist(),
                    key="1v1_jugador_2_select"
                )
            submit_jugadores = st.form_submit_button("Confirmar jugadores")
        if submit_jugadores:
            if jugador_1_label == jugador_2_label:
                st.warning("Debes seleccionar dos jugadores distintos.")
            else:
                # Mapear etiqueta limpia → etiqueta original para mantener la lógica interna
                jugador_1_display = label_to_orig.get(jugador_1_label, jugador_1_label)
                jugador_2_display = label_to_orig.get(jugador_2_label, jugador_2_label)
                st.session_state['1v1_jugador_1_display'] = jugador_1_display
                st.session_state['1v1_jugador_2_display'] = jugador_2_display
                st.session_state['1v1_torneos_1'] = None
                st.session_state['1v1_torneos_2'] = None
                st.session_state['1v1_step'] = 2
                st.rerun()
    else:
        st.success(f"Seleccionaste: **{st.session_state['1v1_jugador_1_display']}** vs **{st.session_state['1v1_jugador_2_display']}**")
        if st.button("🔄 Cambiar jugadores"):
            reset_jugadores()
            st.rerun()

# Paso 2: Selección de torneos para cada jugador
if st.session_state['1v1_step'] == 2 or st.session_state['1v1_torneos_1'] is None or st.session_state['1v1_torneos_2'] is None:
    jugador_1 = st.session_state['1v1_jugador_1_display'] or "Jugador 1"
    jugador_2 = st.session_state['1v1_jugador_2_display'] or "Jugador 2"
    expander_title_torneos = "2️⃣ Confirmar torneos"
else:
    jugador_1 = st.session_state['1v1_jugador_1_display'].split()[0]
    jugador_2 = st.session_state['1v1_jugador_2_display'].split()[0]
    expander_title_torneos = "2️⃣ Confirmar torneos"

if st.session_state['1v1_jugador_1_display'] and st.session_state['1v1_jugador_2_display']:
    df_jug_1 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_1_display']]
    if df_jug_1.empty:
        st.warning("El jugador 1 seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugadores()
        st.stop()
    jugador_1_row = df_jug_1.iloc[0]

    df_jug_2 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_2_display']]
    if df_jug_2.empty:
        st.warning("El jugador 2 seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugadores()
        st.stop()
    jugador_2_row = df_jug_2.iloc[0]

    # Nombres seguros para UI (prefiere ID_Display; si no existe, cae a Nombre_transfermarket)
    j1_name = str(jugador_1_row.get('ID_Display', jugador_1_row.get('Nombre_transfermarket', 'Jugador 1')))
    j2_name = str(jugador_2_row.get('ID_Display', jugador_2_row.get('Nombre_transfermarket', 'Jugador 2')))

    torneos_disp_1 = df_all[df_all["ID"] == jugador_1_row["ID"]]["Torneo"].dropna().tolist()
    torneos_disp_2 = df_all[df_all["ID"] == jugador_2_row["ID"]]["Torneo"].dropna().tolist()
    torneos_disp_1 = sorted(set(t for sublist in torneos_disp_1 for t in (sublist if isinstance(sublist, list) else [sublist])))
    torneos_disp_2 = sorted(set(t for sublist in torneos_disp_2 for t in (sublist if isinstance(sublist, list) else [sublist])))

    with st.expander(expander_title_torneos, expanded=True):
        if st.session_state['1v1_step'] == 2 or st.session_state['1v1_torneos_1'] is None or st.session_state['1v1_torneos_2'] is None:
            with st.form("seleccion_torneos"):
                col1, col2 = st.columns(2)
                with col1:
                    torneos_sel_1 = st.multiselect(
                        f"Torneos de {j1_name}",
                        torneos_disp_1,
                        default=torneos_disp_1,
                        key="1v1_torneos_1_select"
                    )
                with col2:
                    torneos_sel_2 = st.multiselect(
                        f"Torneos de {j2_name}",
                        torneos_disp_2,
                        default=torneos_disp_2,
                        key="1v1_torneos_2_select"
                    )
                submit_torneos = st.form_submit_button("Confirmar torneos")
            if submit_torneos:
                # Guardar torneos confirmados
                st.session_state['1v1_torneos_1'] = torneos_sel_1
                st.session_state['1v1_torneos_2'] = torneos_sel_2

                # --- Overrides para sidebar según Jugador 1 ---
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
                    # Torneos seleccionados de Jugador 1
                    st.session_state['1v1_sync_torneos'] = torneos_sel_1

                    # Posición general de Jugador 1 (derivada de su scope o, si no, del total en df_all)
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
            st.success(
                f"Torneos seleccionados:\n\n- **{j1_name}**: {', '.join(st.session_state['1v1_torneos_1'])}\n"
                f"- **{j2_name}**: {', '.join(st.session_state['1v1_torneos_2'])}"
            )
            if st.button("🔄 Cambiar torneos"):
                reset_torneos()
                st.rerun()

# -------------------- BLOQUE 5: AGREGACIÓN Y FICHA TÉCNICA DE JUGADORES --------------------
# Paso 3: Si todo confirmado, resto de la lógica
if st.session_state['1v1_step'] == 3:
    # Obtener SIEMPRE los objetos correctos (Series) para jugador_1_row y jugador_2_row
    df_jug_1 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_1_display']]
    if df_jug_1.empty:
        st.warning("El jugador 1 seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugadores()
        st.stop()
    jugador_1_row = df_jug_1.iloc[0]

    df_jug_2 = ids_disponibles[ids_disponibles["ID_Display"] == st.session_state['1v1_jugador_2_display']]
    if df_jug_2.empty:
        st.warning("El jugador 2 seleccionado ya no está disponible en los datos actuales. Por favor, vuelve a seleccionarlo.")
        reset_jugadores()
        st.stop()
    jugador_2_row = df_jug_2.iloc[0]
    torneos_1 = st.session_state['1v1_torneos_1']
    torneos_2 = st.session_state['1v1_torneos_2']

    # -------------------- BLOQUE 5A (actualizado): construir muestra final y agregar --------------------
    def extraer_muestra_jugador(df_base: pd.DataFrame, jugador_id, torneos: list) -> pd.DataFrame:
        df_b = df_base.copy()
        df_b["Torneo"] = df_b["Torneo"].apply(lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else []))
        jug_id_str = str(jugador_id)
        return df_b[(df_b["ID"].astype(str) == jug_id_str) & (df_b["Torneo"].apply(lambda ts: any(t in ts for t in torneos)))]

    # Partimos de la muestra filtrada (df) y añadimos los registros de ambos jugadores según torneos elegidos
    df_union = df.copy()
    df_j1_extra = extraer_muestra_jugador(df_all, jugador_1_row["ID"], torneos_1)
    df_j2_extra = extraer_muestra_jugador(df_all, jugador_2_row["ID"], torneos_2)
    if not df_j1_extra.empty:
        df_union = pd.concat([df_union, df_j1_extra], ignore_index=True)
    if not df_j2_extra.empty:
        df_union = pd.concat([df_union, df_j2_extra], ignore_index=True)

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

    # Fichas de jugadores agregadas (para siguiente etapa)
    j1_agg = df_muestra_proc.loc[df_muestra_proc["ID"].astype(str) == str(jugador_1_row["ID"])]
    j2_agg = df_muestra_proc.loc[df_muestra_proc["ID"].astype(str) == str(jugador_2_row["ID"])]
    if j1_agg.empty or j2_agg.empty:
        st.warning("No se pudo preparar la muestra agregada para alguno de los jugadores. Revisa los torneos seleccionados.")
        st.stop()

    jugador_1 = j1_agg.iloc[0]
    jugador_2 = j2_agg.iloc[0]

    # Dejar lista la base procesada para pasos siguientes
    df_agg = df_muestra_proc.copy()

    # -------------------- BLOQUE 5B: Ficha técnica de los jugadores seleccionados --------------------
    st.markdown("---")
    st.markdown("### Ficha técnica de los jugadores seleccionados")
    # -- Estilos modernos para tarjeta de jugador (ligero y responsivo)
    st.markdown(
        """
        <style>
        .player-card{display:flex;gap:16px;padding:14px 16px;border-radius:14px;
                      background:linear-gradient(180deg,rgba(0,0,0,.04),rgba(0,0,0,.02));
                      border:1px solid rgba(0,0,0,.08);box-shadow:0 2px 8px rgba(0,0,0,.06);position:relative;margin-bottom:12px}
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
    col1, col2 = st.columns(2)

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

    with col1:
        mostrar_ficha(jugador_1)
    with col2: 
        mostrar_ficha(jugador_2)

    # -------------------- BLOQUE 6: COMPARATIVA - ELECCIÓN DE MUESTRA --------------------
    # Usar SIEMPRE la muestra filtrada del sidebar
    df_muestra = df.copy()

    # Excluir jugadores seleccionados de la muestra para percentiles comparativos
    df_muestra = df_muestra[~df_muestra["ID"].isin([jugador_1["ID"], jugador_2["ID"]])]

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

    # Info estilo página 1 (4 líneas) - con saltos de línea markdown forzados (2 espacios + \n)
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
    # Esta sección permite seleccionar las métricas a comparar, agrupadas por bloques relevantes
    st.markdown("---")
    st.markdown("### Comparativa de jugadores en base a percentiles")

    # --- Selector avanzado de métricas por tipo relevante (idéntico a página 4) ---
    # Determinar tipos y defaults por posición
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
        # Ya no rellenar con otras métricas si sugeridas está vacío
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
    # Usaremos la misma preparación de métricas, pero aplicada a la muestra del sidebar (excluye a los dos jugadores)
    df_comparacion, _, _, _, metricas_invertir_ctx = aplicar_metricas_personalizadas(df_muestra.copy(), df_metricas)

    # Definir el título y subtítulo del gráfico de radar
    titulo_grafico = (
        f"{jugador_1['Nombre_transfermarket']} {jugador_1['Temporada']} ({jugador_1['Equipo_data']}) "
        f"vs {jugador_2['Nombre_transfermarket']} {jugador_2['Temporada']} ({jugador_2['Equipo_data']})"
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
    temporada_txt = ", ".join(sorted(_temps)) if _temps else str(temporada)

    subtitulo_grafico = (
        f"Comparativa vs {', '.join(pos_det_dimin) if pos_det_dimin else 'muestra filtrada'} "
        f"de {', '.join(torneos)} en {temporada_txt}"
    )

    # Limitar el número de métricas seleccionadas para los gráficos radar
    max_metricas = 12
    if len(seleccionadas) > max_metricas:
        seleccionadas = seleccionadas[:max_metricas]

    # --- Ordenar métricas por bloques (bloque con más seleccionadas primero) y A→Z dentro de cada bloque ---
    # "bloques_metricas_wizard" mantiene el orden de creación; lo usamos como desempate estable
    _bloque_idx = {bn: i for i, bn in enumerate(bloques_metricas_wizard.keys())}

    bloques_para_orden = []
    for nombre_bloque, mets_bloque in bloques_metricas_wizard.items():
        en_bloque = [m for m in seleccionadas if m in mets_bloque]
        if en_bloque:
            en_bloque = sorted(en_bloque, key=_sort_key_az)
            bloques_para_orden.append((nombre_bloque, en_bloque))

    # Métricas que no quedaron clasificadas en ningún bloque (por seguridad)
    resto = [m for m in seleccionadas if not any(m in lst for _, lst in bloques_para_orden)]
    if resto:
        bloques_para_orden.append(("Otros", sorted(resto, key=_sort_key_az)))

    # Ordenar bloques por cantidad (desc) y, en caso de empate, por su orden de creación
    bloques_para_orden.sort(key=lambda x: (-len(x[1]), _bloque_idx.get(x[0], 999)))

    # Aplanar respetando el orden de bloques y A→Z interno
    seleccionadas_radar = []
    for _, lista_mets in bloques_para_orden:
        seleccionadas_radar.extend(lista_mets)

    # Aplicar tope final
    if len(seleccionadas_radar) > max_metricas:
        seleccionadas_radar = seleccionadas_radar[:max_metricas]
    st.info(f"🔢 Máximo de métricas permitidas : {max_metricas}")
    # Calcular percentiles contexto
    percentiles_contexto = {}
    for metrica in seleccionadas:
        serie = df_comparacion[metrica].replace(0, np.nan).dropna()
        percentiles_contexto[metrica] = {
            idx: rankdata(serie, method='average')[i] / len(serie)
            for i, idx in enumerate(serie.index)
        }
    # Cálculo promedio muestra
    promedio_valores = df_comparacion[seleccionadas].replace(0, np.nan).mean().to_frame().T
    promedio_valores['Nombre_transfermarket'] = 'Promedio muestra'
    promedio_valores['Equipo_data'] = ''
    promedio_valores['Temporada'] = ''
    promedio_valores['ID'] = 'PROMEDIO'
    promedio_valores.index = [999]
    percentiles_contexto.update({
        metrica: {**percentiles_contexto.get(metrica, {}), 999: rankdata(df_comparacion[metrica].replace(0, np.nan).dropna().tolist() + [promedio_valores.iloc[0][metrica]], method='average')[-1] / (len(df_comparacion) + 1)}
        for metrica in seleccionadas
    })
    # Forzar el modo según el tema actual
    modo_claro = (get_theme_type() == "light")
    fig = crear_radar_percentil_plotly(
        [jugador_1, jugador_2],
        [f"{jugador_1['Nombre_transfermarket']} - {jugador_1['Equipo_data']} - {jugador_1['Temporada']}",
            f"{jugador_2['Nombre_transfermarket']} - {jugador_2['Equipo_data']} - {jugador_2['Temporada']}"],
        seleccionadas_radar,
        df_comparacion,
        titulo=titulo_grafico,
        subtitulo=None,
        modo_claro=modo_claro
    )
    col1 = st.columns(1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="radar_chart", theme=None) 
    # Información de la muestra centrada bajo la gráfica
    def _center_caption(md_text: str) -> str:
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md_text)
        html = html.replace("\n", "<br>")
        return f"<div style='text-align:center; font-size:0.9rem; opacity:0.85'>{html}</div>"

    st.markdown(_center_caption(info_md), unsafe_allow_html=True)
    col2 = st.columns(1)

    # Nombres de columnas
    nombre_1 = f"{jugador_1['Nombre_transfermarket']} - {jugador_1['Temporada']}"
    nombre_2 = f"{jugador_2['Nombre_transfermarket']} - {jugador_2['Temporada']}"
    col_perc_1 = "Percentil 1"
    col_perc_2 = "Percentil 2"

    # Datos
    datos_tabla = []
    for metrica in seleccionadas_radar:
        val_1 = jugador_1[metrica]
        val_fmt_1 = formatear_valor(metrica, val_1)
        perc_1 = min(max(obtener_percentiles(jugador_1, df_comparacion, [metrica], metricas_invertir_ctx)[0], 0), 100)

        val_2 = jugador_2[metrica]
        val_fmt_2 = formatear_valor(metrica, val_2)
        perc_2 = min(max(obtener_percentiles(jugador_2, df_comparacion, [metrica], metricas_invertir_ctx)[0], 0), 100)

        datos_tabla.append({
            "Métrica": metrica,
            nombre_1: val_fmt_1,
            col_perc_1: perc_1,
            nombre_2: val_fmt_2,
            col_perc_2: perc_2
        })

    # Respetar el orden de `seleccionadas_radar` (por bloques y A→Z)
    df_tabla = pd.DataFrame(datos_tabla)
    # Forzar el orden de filas según 'seleccionadas_radar' (bloques y A→Z)
    try:
        df_tabla["Métrica"] = pd.Categorical(
            df_tabla["Métrica"],
            categories=seleccionadas_radar,
            ordered=True
        )
        df_tabla = df_tabla.sort_values("Métrica").reset_index(drop=True)
    except Exception:
        # Fallback robusto por si alguna métrica no está en la lista
        orden_map = {m: i for i, m in enumerate(seleccionadas_radar)}
        df_tabla["__orden__"] = df_tabla["Métrica"].map(orden_map).fillna(1e9)
        df_tabla = df_tabla.sort_values("__orden__").drop(columns="__orden__").reset_index(drop=True)

    # --- Tabla moderna (alternativa a AgGrid): valores y percentiles por métrica (comparativa)
    st.markdown("#### Tabla moderna: valores y percentiles por métrica (comparativa)")

    # CSS (ligero, funciona en claro/oscuro)
    tbl_css_cmp = """
    <style>
    .table-modern{width:100%;border-collapse:separate;border-spacing:0 8px;table-layout:fixed}
    .table-modern th{font-size:.85rem;text-align:center;opacity:.85;padding:6px 10px}
    .table-modern td{background:rgba(0,0,0,.03);padding:10px;border:1px solid rgba(0,0,0,.06)}
    .table-modern td:first-child{border-top-left-radius:10px;border-bottom-left-radius:10px}
    .table-modern td:last-child{border-top-right-radius:10px;border-bottom-right-radius:10px}
    .badge-metric{font-weight:600}
    .td-center{text-align:center}
    .metric-center{text-align:center;font-weight:600}

    /* Progress bars */
    .pbar-wrap{display:flex;align-items:center;gap:10px}
    .pbar-wrap-rtl{display:flex;align-items:center;gap:10px}
    .pbar{flex:1;height:10px;border-radius:999px;background:rgba(0,0,0,.08);overflow:hidden}
    .pbar-rtl{transform:scaleX(-1);transform-origin:center}
    .pbar>span{display:block;height:100%;border-radius:inherit}
    .pct-label{width:42px;text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
    .pct-label-left{width:42px;text-align:left;font-variant-numeric:tabular-nums;font-weight:700}

    /* Grouped headers */
    .th-group{font-weight:700}
    .th-metric{font-weight:700}
    .th-sub{font-weight:600;opacity:.8}

    /* Fixed column widths */
    .table-modern col.c1{width:10%}
    .table-modern col.c2{width:24%}
    .table-modern col.c3{width:32%}
    .table-modern col.c4{width:24%}
    .table-modern col.c5{width:10%}

    @media (prefers-color-scheme: dark){
      .table-modern td{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.08)}
      .pbar{background:rgba(255,255,255,.12)}
    }
    </style>
    """

    def _bar_html_from_pct(pct):
        if pct is None:
            return "<div class='pbar'><span style='width:0%'></span></div>", "N/D"
        try:
            p = float(pct)
        except Exception:
            p = 0.0
        p = max(0.0, min(100.0, p))
        hue = int(round((p/100.0)*120))  # 0=rojo, 120=verde
        bar = f"<div class='pbar'><span style='width:{p}%;background:hsl({hue},70%,45%);'></span></div>"
        lab = f"{int(round(p))}%"
        return bar, lab

    def _bar_html_from_pct_rtl(pct):
        if pct is None:
            return "<div class='pbar pbar-rtl'><span style='width:0%'></span></div>", "N/D"
        try:
            p = float(pct)
        except Exception:
            p = 0.0
        p = max(0.0, min(100.0, p))
        hue = int(round((p/100.0)*120))  # 0=rojo, 120=verde
        bar = f"<div class='pbar pbar-rtl'><span style='width:{p}%;background:hsl({hue},70%,45%);'></span></div>"
        lab = f"{int(round(p))}%"
        return bar, lab

    rows_html_cmp = []
    for r in df_tabla.to_dict(orient="records"):
        met = _sanitize_text(r.get("Métrica", ""))
        v1_fmt = _sanitize_text(r.get(nombre_1, ""))
        v2_fmt = _sanitize_text(r.get(nombre_2, ""))
        p1    = r.get(col_perc_1, None)
        p2    = r.get(col_perc_2, None)

        bar1_rtl, lab1 = _bar_html_from_pct_rtl(p1 if (p1 is not None and pd.notna(p1)) else None)
        bar2,     lab2 = _bar_html_from_pct(p2 if (p2 is not None and pd.notna(p2)) else None)

        # Determinar ganador según el valor real (considerando métricas invertidas)
        try:
            v1_num = float(jugador_1.get(met, np.nan))
        except Exception:
            v1_num = np.nan
        try:
            v2_num = float(jugador_2.get(met, np.nan))
        except Exception:
            v2_num = np.nan

        lower_better = isinstance(metricas_invertir_ctx, (set, list, dict)) and (met in metricas_invertir_ctx)
        winner = None
        if not np.isnan(v1_num) and not np.isnan(v2_num):
            if lower_better:
                if v1_num < v2_num: winner = 1
                elif v2_num < v1_num: winner = 2
            else:
                if v1_num > v2_num: winner = 1
                elif v2_num > v1_num: winner = 2

        win1 = "<span class='win'> 🔼</span>" if winner == 1 else ""
        win2 = "<span class='win'> 🔼</span>" if winner == 2 else ""

        rows_html_cmp.append(
            "<tr>"
            # 1) Valor Jugador 1 (centrado)
            f"<td class='td-center'>{v1_fmt}{win1}</td>"
            # 2) Percentil Jugador 1 (label a la izquierda, barra de der→izq)
            f"<td><div class='pbar-wrap-rtl'><div class='pct-label-left'>{lab1}</div>{bar1_rtl}</div></td>"
            # 3) Métrica (centrada)
            f"<td class='metric-center'>{met}</td>"
            # 4) Percentil Jugador 2 (tal cual estaba: barra + label a la derecha)
            f"<td><div class='pbar-wrap'>{bar2}<div class='pct-label'>{lab2}</div></div></td>"
            # 5) Valor Jugador 2 (tal cual + icono)
            f"<td>{v2_fmt}{win2}</td>"
            "</tr>"
        )

    colgroup_html = (
        "<colgroup>"
        "<col class='c1'>"
        "<col class='c2'>"
        "<col class='c3'>"
        "<col class='c4'>"
        "<col class='c5'>"
        "</colgroup>"
    )

    header_html = (
        "<thead>"
        "<tr>"
        f"<th colspan='2' class='th-group'>{_sanitize_text(nombre_1)}</th>"
        "<th rowspan='2' class='th-metric'>Métrica</th>"
        f"<th colspan='2' class='th-group'>{_sanitize_text(nombre_2)}</th>"
        "</tr>"
        "<tr>"
        "<th class='th-sub'>Valor</th>"
        "<th class='th-sub'>Percentil</th>"
        "<th class='th-sub'>Percentil</th>"
        "<th class='th-sub'>Valor</th>"
        "</tr>"
        "</thead>"
    )

    tabla_cmp_html = (
        tbl_css_cmp +
        "<table class='table-modern'>" +
        colgroup_html +
        header_html +
        "<tbody>" + "".join(rows_html_cmp) + "</tbody></table>"
    )
    st.markdown(tabla_cmp_html, unsafe_allow_html=True)

    # Aclaración breve sobre percentiles
    st.markdown(
        """
        *Nota:* el **percentil** indica la posición relativa dentro de la muestra. Por ejemplo,
        estar en el **percentil 99** significa que el jugador está igual o por encima del 99% de los
        jugadores de la muestra en esa métrica (solo ~1% lo supera). Un **percentil 50** es la mediana.
        """
    )



# ========================================================================
# -------------------- BLOQUE 8: Radares comparativos por bloque (3x2) --------------------
# ========================================================================

    st.markdown("---")
    st.markdown("### Radares comparativos por bloque")

    # Helper: elegir métricas del universo del bloque (no sólo seleccionadas)
    # - Respeta el modo (Por 90' / Totales) usando considerar_dict
    # - Prioriza métricas sugeridas por posición detallada del Jugador 1
    # - Limita a un máximo razonable por radar para legibilidad

    def _elegir_metricas_bloque(univ_bloque: list[str], considerar_dict: dict, modo_90: bool,
                                df_cmp: pd.DataFrame, pos_det_ref: str, max_mets: int = 12) -> list[str]:
        if not univ_bloque:
            return []
        tipos_validos = ["/90", "Porcentaje"] if modo_90 else ["Totales", "Porcentaje"]
        # Filtrar por tipo válido y existencia en df
        candidatas = [m for m in univ_bloque if (m in considerar_dict) and (considerar_dict[m] in tipos_validos) and (m in df_cmp.columns)]
        if not candidatas:
            return []
        # Prioridad: métricas default por posición detallada
        try:
            defaults_pos = metricas_default_por_posicion.get(pos_det_ref, [])
        except Exception:
            defaults_pos = []
        pri = [m for m in candidatas if m in defaults_pos]
        resto = sorted([m for m in candidatas if m not in defaults_pos], key=_sort_key_az)
        orden = pri + resto
        if len(orden) > max_mets:
            orden = orden[:max_mets]
        # Asegurar al menos 3 para que el radar sea informativo
        return orden if len(orden) >= 3 else []

    # Definir bloques por tipo de jugador
    pos_det_j1 = jugador_1.get("Posicion_detallada", "")
    diminutivo_pos_det = diminutivos_pos.get(pos_det_j1, pos_det_j1)

    if diminutivo_pos_det == "PT":
        bloques = [
            ("Portero",                 metricas_portero),
            ("Físicas",                 metricas_fisicas),
            ("Construcción (general)",  metricas_construccion_general),
        ]
    else:
        bloques = [
            ("Físicas",                  metricas_fisicas),
            ("Construcción (general)",   metricas_construccion_general),
            ("Construcción (ofensiva)",  metricas_construccion_ofensiva),
            ("Centros",                  metricas_centros),
            ("Ofensivas",                metricas_ofensivas),
            ("Balón parado",             metricas_balon_parado),
        ]

    # Preparar lista final de (nombre_bloque, métricas)
    valid_blocks: list[tuple[str, list[str]]] = []
    for nb, lst in bloques:
        mets = _elegir_metricas_bloque(lst, considerar_dict, modo_90, df_comparacion, pos_det_j1, max_mets=12)
        if mets:
            valid_blocks.append((nb, mets))

    if not valid_blocks:
        st.info("No hay bloques con métricas suficientes para el modo actual.")
    else:
        # Etiquetas para ambos jugadores
        labels_duo = [
            f"{jugador_1['Nombre_transfermarket']} - {jugador_1['Equipo_data']} - {jugador_1['Temporada']}",
            f"{jugador_2['Nombre_transfermarket']} - {jugador_2['Equipo_data']} - {jugador_2['Temporada']}"
        ]
        modo_claro = (get_theme_type() == "light")

        # Render en grilla: 3 filas x 2 columnas (hasta 6 bloques)
        # Se imprimen en pares; si hay menos de 6, se llenan en orden
        for i in range(0, min(len(valid_blocks), 6), 2):
            c1, c2 = st.columns(2, gap="large")
            nb1, mets1 = valid_blocks[i]
            with c1:
                fig_b1 = crear_radar_percentil_plotly(
                    [jugador_1, jugador_2],
                    labels_duo,
                    mets1,
                    df_comparacion,
                    titulo=f"{nb1}",
                    subtitulo=None,
                    modo_claro=modo_claro,
                )
                st.plotly_chart(fig_b1, use_container_width=True, config={"displayModeBar": False}, theme=None)
            if i + 1 < len(valid_blocks):
                nb2, mets2 = valid_blocks[i+1]
                with c2:
                    fig_b2 = crear_radar_percentil_plotly(
                        [jugador_1, jugador_2],
                        labels_duo,
                        mets2,
                        df_comparacion,
                        titulo=f"{nb2}",
                        subtitulo=None,
                        modo_claro=modo_claro,
                    )
                    st.plotly_chart(fig_b2, use_container_width=True, config={"displayModeBar": False}, theme=None)