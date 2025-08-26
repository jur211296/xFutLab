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


import re


# --- Sanitizador simple para textos de hover/etiquetas (evita caracteres de control) ---
_re_bad = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")

def _sanitize_text(val) -> str:
    try:
        return _re_bad.sub("", str(val))
    except Exception:
        return str(val)

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
        aplicar_metricas_personalizadas, diminutivos_pos, formatear_valor
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
st.sidebar.markdown("## ⚙️ Filtros generales")

 # --- Temporada ---
temporadas_disponibles = sorted(df["Temporada"].dropna().unique(), reverse=True) if "Temporada" in df.columns else []
if not temporadas_disponibles:
    st.error("No hay temporadas disponibles en los datos.")
    st.stop()

default_temporada = next((t for t in temporadas_disponibles if "2025" in str(t)), temporadas_disponibles[0])
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
paises_sel = st.sidebar.multiselect(
    "País del equipo",
    paises_opciones,
    default=([default_pais] if default_pais else [])
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

torneos_sel = st.sidebar.multiselect(
    "Torneo",
    torneos_opciones,
    default=torneos_opciones
)

# --- Equipo ---
if "Equipo_data" in df_temp.columns:
    # Equipos disponibles considerando temporada + país + torneo
    if paises_sel and torneos_sel:
        mask_eq = df_temp["Pais"].isin(paises_sel) & df_temp["Torneo"].isin(torneos_sel)
    elif paises_sel:
        mask_eq = df_temp["Pais"].isin(paises_sel)
    elif torneos_sel:
        mask_eq = df_temp["Torneo"].isin(torneos_sel)
    else:
        mask_eq = pd.Series(True, index=df_temp.index)
    equipos_opciones = sorted(df_temp.loc[mask_eq, "Equipo_data"].dropna().unique())
    equipos_sel = st.sidebar.multiselect(
        "Equipo",
        equipos_opciones,
        default=equipos_opciones
    )
else:
    equipos_sel = []

# DF filtrado preliminar (pais/torneo/equipo) – hacemos copy para evitar SettingWithCopy
mask_base = df_temp["Pais"].isin(paises_sel) & df_temp["Torneo"].isin(torneos_sel)
if "Equipo_data" in df_temp.columns and equipos_sel:
    mask_base &= df_temp["Equipo_data"].isin(equipos_sel)
df_filtros = df_temp[mask_base].copy()


# -------------------- FILTROS RESTANTES EN EXPANDER --------------------
with st.expander("🔍 Filtros de jugadores a analizar", expanded=True):
    st.markdown("### :mag_right: Filtros")

    if df_filtros.empty:
        st.warning("No hay datos para los filtros actuales de País/Torneo.")
        st.stop()

    col4, col5, col6 = st.columns(3)
    with col4:
        min_jugados_max = int(df_filtros["Minutos_jugados"].max()) if "Minutos_jugados" in df_filtros.columns else 0
        min_jugados = st.slider("Minutos jugados (mínimo)", 0, max(min_jugados_max, 0), min(300, max(min_jugados_max, 0)), step=50)
    with col5:
        max_m90 = int(df_filtros["M90s_jugados"].max()) if "M90s_jugados" in df_filtros.columns else 0
        min_m90s = st.slider("Partidos completos jugados (M90s)", 0, max(max_m90, 0), min(3, max(max_m90, 0)), step=1)
    with col6:
        if "Edad" in df_filtros.columns:
            edad_min = int(df_filtros["Edad"].min())
            edad_max = int(df_filtros["Edad"].max())
            edad_range = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
        else:
            edad_range = (0, 100)

    col7, col8, col9 = st.columns(3)
    with col7:
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

        # Default: todas menos Portero si existe
        default_pos_gen = [p for p in posiciones_gen if p != "Portero"] if posiciones_gen else []

        # Opciones visuales (abreviaturas) y defaults visuales
        opciones_vis = [abbr_map[p] for p in posiciones_gen]
        default_vis = [abbr_map[p] for p in default_pos_gen]

        _seg = getattr(st, "segmented_control", None)
        if callable(_seg):
            try:
                sel_vis = _seg("Posición general", opciones_vis, selection_mode="multi", default=default_vis)
            except Exception:
                sel_vis = _seg("Posición general", opciones_vis, default=default_vis)
                if isinstance(sel_vis, str):
                    sel_vis = [sel_vis]
        else:
            sel_vis = st.multiselect("Posición general", opciones_vis, default=default_vis)

        if isinstance(sel_vis, str):
            sel_vis = [sel_vis]
        # Convertir abreviaturas a labels reales usadas en el dataset
        pos_gen_sel = [rev_abbr.get(lbl, lbl) for lbl in sel_vis]
    with col8:
        if pos_gen_sel and "Posicion_detallada" in df_filtros.columns:
            _raw_det = df_filtros[df_filtros["Posicion_general"].isin(pos_gen_sel)]["Posicion_detallada"].dropna()
            _raw_det = _raw_det.astype(str)
            _bad = {"nan", "none", "null", "", "na"}
            _raw_det = _raw_det[~_raw_det.str.strip().str.lower().isin(_bad)]
            pos_det_opciones = sorted(_raw_det.unique())
        else:
            pos_det_opciones = []
        pos_det_sel = st.multiselect("Posición detallada", pos_det_opciones, default=pos_det_opciones)

    usar_nacionalidad_detallada = st.checkbox("Cambiar a nacionalidad detallada", value=False)

    with col9:
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
    ((df_temp["Equipo_data"].isin(equipos_sel)) if ("Equipo_data" in df_temp.columns and equipos_sel) else True) &
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

# ========================================================================
# -------------------- BLOQUE 4: AGRUPACIÓN Y CÁLCULO DE MÉTRICAS --------------------
# ========================================================================
# Agrupa si hay más de un torneo, aplica métricas personalizadas y columnas extra
if "Torneo" in df.columns and df["Torneo"].nunique() > 1:
    # Sumar solo columnas numéricas (evitar Edad)
    columnas_sumables = [
        c for c in df.columns
        if is_numeric_dtype(df[c]) and c not in ["Edad"]
    ]
    agg_spec = {col: "sum" for col in columnas_sumables}

    # Tomar la primera aparición para columnas categóricas/identitarias si existen
    opcionales_first = [
        "Nombre_transfermarket", "ID_Equipo", "logo_equipo", "Equipo_data",
        "Posicion_general", "Posicion_detallada", "Pais",
        "Pais_diminutivo", "Nacionalidad_2", "Nacionalidad",
        "Edad", "Temporada", "Color primario"
    ]
    for col in opcionales_first:
        if col in df.columns:
            agg_spec[col] = "first"

    # Torneo → lista única de torneos por ID (si existe)
    if "Torneo" in df.columns:
        agg_spec["Torneo"] = lambda x: list(set(x.dropna()))

    df_agg = df.groupby("ID", as_index=False).agg(agg_spec)
else:
    df_agg = df.copy()

# Aplicar métricas personalizadas, /90, renombrados, diccionarios
df_agg, considerar_dict, tipos_dict, metricas_porcentaje, metricas_invertir = aplicar_metricas_personalizadas(df_agg, df_metricas)

# Columna compuesta de equipo y país
if all(col in df_agg.columns for col in ["Equipo_data", "Pais_diminutivo"]):
    df_agg["Equipo_data_full"] = df_agg["Equipo_data"] + " " + df_agg["Pais_diminutivo"]

# Cachear columnas disponibles para filtrar métricas y resolver nombres
available_cols = set(df_agg.columns)

# ========================================================================
# -------------------- BLOQUE 5: PLACEHOLDER (SIN VISUALIZACIONES) --------------------
# ========================================================================
# ======================
# Bloque 5: Encabezado final (sin vistas previas)
# ======================

# Helpers para formatear listas con regla "mostrar hasta N" o etiquetar como múltiples
def _join_or_multiple(vals, max_show, multiple_label):
    vals = [str(v) for v in vals if v is not None]
    if not vals:
        return "N/A"
    return ", ".join(vals) if len(vals) <= max_show else multiple_label

# Cálculos para el mensaje de "Datos preparados"
jugadores_unicos = int(df["ID"].nunique()) if "ID" in df.columns else len(df)

# País del equipo: si son <3, listar; si no, "Múltiples países"
paises_text = _join_or_multiple(paises_sel, 2, "Múltiples países")

# Torneo: si son <5, listar; si no, "Múltiples torneos"
torneos_text = _join_or_multiple(torneos_sel, 4, "Múltiples torneos")

# Posición general: si seleccionaron todas las disponibles -> "Todas"
if 'posiciones_gen' in locals() and pos_gen_sel and posiciones_gen:
    pos_gen_text = "Todas" if set(pos_gen_sel) == set(posiciones_gen) else ", ".join(map(str, pos_gen_sel))
else:
    pos_gen_text = ", ".join(map(str, pos_gen_sel)) if pos_gen_sel else "N/A"

# Posición detallada: si seleccionaron todas las opciones disponibles -> "Todas"
if 'pos_det_opciones' in locals() and pos_det_sel and pos_det_opciones:
    pos_det_text = "Todas" if set(pos_det_sel) == set(pos_det_opciones) else ", ".join(map(str, pos_det_sel))
else:
    pos_det_text = ", ".join(map(str, pos_det_sel)) if pos_det_sel else "N/A"

# Nacionalidad: si es detallada y hay más de 3 seleccionadas, mostrar "Múltiples nacionalidades"
if usar_nacionalidad_detallada:
    nac_text = ", ".join(map(str, nac2_sel)) if (nac2_sel and len(nac2_sel) <= 3) else ("Múltiples nacionalidades" if nac2_sel else "N/A")
else:
    nac_text = ", ".join(map(str, nac2_sel)) if nac2_sel else "N/A"

# Mensaje de datos preparados ANTES del título principal
st.info(
    (
        f"La muestra filtrada contiene **{jugadores_unicos} jugadores únicos**.  \n"
        f"**Temporada:** {temporada} - **Pais del equipo:** {paises_text} - **Torneo:** {torneos_text}  \n"
        f"**Minutos jugados:** >{min_jugados} - **Partidos completos jugados:** >{min_m90s} - **Edad:** {int(edad_range[0])}-{int(edad_range[1])} - **Nacionalidad:** {nac_text}.  \n"
        f"**Posición general:** {pos_gen_text} - **Posición detallada:** {pos_det_text}"
    )
)

# Título y única leyenda inferior (fecha de actualización)
st.markdown("## Comparativa general")

# Crear relaciones entre métricas Totales y /90 (por nombre en app), solo si existen en el DF
relacion_metricas_totales_90 = {}   # 'Asistencias esperadas' -> 'Asistencias esperadas /90'
relacion_metricas_90_totales = {}   # 'Asistencias esperadas /90' -> 'Asistencias esperadas'
for _, row in df_metricas.iterrows():
    app_name = row.get("Nombre en app")
    considerar = row.get("Considerar")
    if isinstance(app_name, str) and considerar == "/90" and app_name.endswith("/90"):
        base_app = app_name.replace(" /90", "").replace("/90", "")
        # Solo registrar relaciones si ambas columnas existen en df_agg
        if base_app in available_cols and app_name in available_cols:
            relacion_metricas_totales_90[base_app] = app_name
            relacion_metricas_90_totales[app_name] = base_app

# Filtrar todas las métricas candidatas SEGÚN columnas disponibles
metricas_totales = df_metricas[df_metricas["Considerar"] == "Totales"]["Nombre en app"].dropna().tolist()
metricas_90 = df_metricas[df_metricas["Considerar"] == "/90"]["Nombre en app"].dropna().tolist()
metricas_perc = df_metricas[df_metricas["Considerar"] == "Porcentaje"]["Nombre en app"].dropna().tolist()

metricas_mostrar_totales = sorted([m for m in (metricas_totales + metricas_perc) if m in available_cols])
metricas_mostrar_90 = sorted([m for m in (metricas_90 + metricas_perc) if m in available_cols])

# --- Validación y selección segura de métricas iniciales ---
if "selected_x" not in st.session_state or st.session_state.selected_x not in metricas_mostrar_totales:
    if metricas_mostrar_totales:
        st.session_state.selected_x = metricas_mostrar_totales[0]
    else:
        st.session_state.selected_x = ""
if "selected_y" not in st.session_state or st.session_state.selected_y not in metricas_mostrar_totales:
    if len(metricas_mostrar_totales) > 1:
        st.session_state.selected_y = metricas_mostrar_totales[1]
    elif metricas_mostrar_totales:
        st.session_state.selected_y = metricas_mostrar_totales[0]
    else:
        st.session_state.selected_y = ""

# Toggle de Totales ↔ /90
modo_90 = st.toggle("Métricas: Totales ↔ Por 90'", value=st.session_state.get("scatter_modo_90", True), key="scatter_modo_90")

# Definir métricas mostradas basadas en toggle
metricas_mostrar = metricas_mostrar_90 if modo_90 else metricas_mostrar_totales

# 🔥 Función para convertir dinámicamente métricas seleccionadas
def convertir_metrica_toggle(metrica, modo_90):
    # Porcentajes no cambian
    if metrica in metricas_perc:
        return metrica

    def has_p90(name: str) -> bool:
        return name.endswith("/90") or name.endswith(" /90")

    if modo_90:
        if has_p90(metrica):
            return metrica
        candidate = relacion_metricas_totales_90.get(metrica)
        if not candidate:
            candidate = f"{metrica} /90"
        return candidate if candidate in metricas_mostrar_90 else metrica
    else:
        if has_p90(metrica):
            candidate = relacion_metricas_90_totales.get(metrica)
            if not candidate:
                candidate = metrica.replace(" /90", "").replace("/90", "")
            return candidate if candidate in metricas_mostrar_totales else metrica
        return metrica

# 🔥 Convertir las seleccionadas actuales dinámicamente
x_metric_convertido = convertir_metrica_toggle(st.session_state.selected_x, modo_90)
y_metric_convertido = convertir_metrica_toggle(st.session_state.selected_y, modo_90)

# Lógica de tipos de métricas según posiciones seleccionadas
if pos_gen_sel == ["Portero"]:
    tipos_disponibles = ["Portero"]
elif "Portero" not in pos_gen_sel:
    tipos_disponibles = [t for t in set(tipos_dict.values()) if t != "Portero"]
else:
    tipos_disponibles = sorted(set(tipos_dict.values()))

with st.expander("🎛️ Parámetros de visualización", expanded=True):
    col_tipo1, col_tipo2 = st.columns(2)
    with col_tipo1:
        tipo_x = st.selectbox("Tipo métrica eje X", sorted(tipos_disponibles), index=0)
    with col_tipo2:
        tipo_y = st.selectbox("Tipo métrica eje Y", sorted(tipos_disponibles), index=1 if len(tipos_disponibles) > 1 else 0)

    metricas_x_candidatas = [m for m in metricas_mostrar if tipos_dict.get(m) == tipo_x]
    metricas_y_candidatas = [m for m in metricas_mostrar if tipos_dict.get(m) == tipo_y]

    # Guard clause: Si no hay métricas candidatas, avisar y detener
    if not metricas_x_candidatas or not metricas_y_candidatas:
        st.warning("No hay métricas disponibles para los tipos seleccionados.")
        st.stop()

    # Validar métricas seleccionadas
    if st.session_state.selected_x not in metricas_x_candidatas:
        st.session_state.selected_x = metricas_x_candidatas[0] if metricas_x_candidatas else ""
    if st.session_state.selected_y not in metricas_y_candidatas:
        st.session_state.selected_y = metricas_y_candidatas[0] if metricas_y_candidatas else ""

    # Validar que las métricas seleccionadas estén en las opciones candidatas, si no, usar la primera
    index_x = metricas_x_candidatas.index(x_metric_convertido) if x_metric_convertido in metricas_x_candidatas else 0
    index_y = metricas_y_candidatas.index(y_metric_convertido) if y_metric_convertido in metricas_y_candidatas else 0

    colx, coly = st.columns(2)
    with colx:
        nueva_x = st.selectbox("Métrica a representar en el eje X", metricas_x_candidatas, index=index_x)
        if nueva_x != st.session_state.selected_x:
            st.session_state.selected_x = nueva_x
    with coly:
        nueva_y = st.selectbox("Métrica a representar en el eje Y", metricas_y_candidatas, index=index_y)
        if nueva_y != st.session_state.selected_y:
            st.session_state.selected_y = nueva_y

    x_metric = st.session_state.selected_x
    y_metric = st.session_state.selected_y

# --- Resolver nombres de métricas ante diferencias mínimas de espacios/sufijos ---
def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s2 = s.replace(" - ", "-")
    s2 = s2.replace(" /90", "/90").replace("/ 90", "/90")
    s2 = " ".join(s2.split())
    return s2.lower()

def resolver_metrica(nombre: str, cols: set[str]) -> str | None:
    if nombre in cols:
        return nombre
    n = _norm_name(nombre)
    for c in cols:
        if _norm_name(c) == n:
            return c
    return None

# === Utilidades para construir subtítulo de posiciones ===

def _join_con_y(items: list[str]) -> str:
    items = [str(x) for x in items if x is not None and str(x) != ""]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} y {items[1]}"
    return ", ".join(items[:-1]) + f" y {items[-1]}"

def construir_linea_posiciones(df_ctx: pd.DataFrame, pos_gen_sel: list[str], pos_det_sel: list[str]) -> str:
    """Genera una línea descriptiva de posiciones en el orden DEF, MED, DEL.
    - Si un grupo tiene todas sus posiciones detalladas seleccionadas → "Todos los ...".
    - Si un grupo tiene selección parcial → "<ABREV>: pos1, pos2".
    """
    orden = [
        ("Portero", "GK", "porteros"),
        ("Defensa", "DEF", "defensas"),
        ("Mediocampista", "MED", "mediocampistas"),
        ("Delantero", "DEL", "delanteros"),
    ]
    
    # Mapeo Posicion_detallada -> Posicion_general desde el contexto filtrado
    _bad = {"nan", "none", "null", "", "na"}
    map_det_a_gen = {}
    if {"Posicion_detallada", "Posicion_general"}.issubset(df_ctx.columns):
        pares = df_ctx[["Posicion_detallada", "Posicion_general"]].dropna().drop_duplicates()
        pares = pares[~pares["Posicion_detallada"].astype(str).str.strip().str.lower().isin(_bad)]
        map_det_a_gen = dict(zip(pares["Posicion_detallada"], pares["Posicion_general"]))

    # Detalladas disponibles por grupo en el contexto actual
    all_by_group: dict[str, list[str]] = {}
    sel_by_group: dict[str, list[str]] = {}

    for g, _, _ in orden:
        if "Posicion_general" in df_ctx.columns and "Posicion_detallada" in df_ctx.columns:
            _tmp = df_ctx.loc[df_ctx["Posicion_general"] == g, "Posicion_detallada"].dropna().astype(str)
            _tmp = _tmp[~_tmp.str.strip().str.lower().isin(_bad)]
            all_g = sorted(_tmp.unique().tolist())
        else:
            all_g = []
        all_by_group[g] = all_g
        # Seleccionadas de ese grupo (limpiando valores inválidos)
        _sel_clean = [str(p) for p in pos_det_sel if str(p).strip().lower() not in _bad]
        sel_g = [p for p in _sel_clean if map_det_a_gen.get(p) == g]
        sel_by_group[g] = sel_g

    # Grupos con “todos” seleccionados (solo si el grupo está dentro de pos_gen_sel)
    grupos_todos = []
    for g, _, plural in orden:
        if g in pos_gen_sel:
            if not all_by_group[g]:
                # Si no hay detalladas registradas para el grupo, interpretar como todos
                grupos_todos.append(f"Todos los {plural}")
            elif set(sel_by_group[g]) == set(all_by_group[g]):
                grupos_todos.append(f"Todos los {plural}")

    primera_parte = _join_con_y(grupos_todos).capitalize() if grupos_todos else ""

    # Grupos con selección parcial → listar detalladas con prefijo abreviado
    partes_parciales = []
    for g, abbr, _ in orden:
        if g in pos_gen_sel and sel_by_group[g] and not (all_by_group[g] and set(sel_by_group[g]) == set(all_by_group[g])):
            partes_parciales.append(f"{abbr}: {', '.join(sel_by_group[g])}.")

    if primera_parte and partes_parciales:
        return f"{primera_parte}. " + " ".join(partes_parciales)
    if primera_parte:
        return primera_parte
    if partes_parciales:
        return " ".join(partes_parciales)
    # Si no hay nada seleccionado específicamente, caer a texto general
    return "Posiciones seleccionadas"

x_metric_res = resolver_metrica(x_metric, available_cols)
y_metric_res = resolver_metrica(y_metric, available_cols)

if not x_metric_res or not y_metric_res:
    # Fallback a la primera opción válida de las listas mostradas
    fallback_x = next((m for m in (metricas_mostrar_90 if modo_90 else metricas_mostrar_totales) if m in available_cols), None)
    fallback_y = next((m for m in (metricas_mostrar_90 if modo_90 else metricas_mostrar_totales) if m in available_cols and m != fallback_x), fallback_x)
    if not fallback_x or not fallback_y:
        st.error("No hay columnas disponibles que coincidan con las métricas seleccionadas.")
        st.stop()
    x_metric, y_metric = fallback_x, fallback_y
else:
    x_metric, y_metric = x_metric_res, y_metric_res

# === Guardias y sincronización de selección de métricas ===
if x_metric == y_metric:
    st.warning("Las métricas de los ejes X e Y no pueden ser iguales. Selecciona métricas distintas.")
    st.stop()

# Guardar en session_state para que otras páginas puedan leerlas
st.session_state["scatter_metric_x"] = x_metric
st.session_state["scatter_metric_y"] = y_metric
# No asignar a \"scatter_modo_90\" porque es un widget con key.

@st.cache_data(show_spinner=False)
def preparar_visualizacion(df, x_metric, y_metric, jitter_strength, metricas_invertir):

    df_viz = df[df[x_metric].notna() & df[y_metric].notna()].copy()
    if x_metric in metricas_invertir:
        df_viz[x_metric] = -df_viz[x_metric]
    if y_metric in metricas_invertir:
        df_viz[y_metric] = -df_viz[y_metric]

    # Sin desempate/jitter: se conservan las coordenadas originales

    relevancia = df_viz[x_metric].rank(pct=True) + df_viz[y_metric].rank(pct=True)
    df_viz["Relevancia"] = relevancia

    hover_series = (
        "Jugador: " + df_viz["Nombre_transfermarket"].astype(str) + "<br>" +
        "Equipo: " + df_viz.get("Equipo_data_full", df_viz.get("Equipo_data", pd.Series("", index=df_viz.index))).astype(str) + "<br>" +
        "Minutos jugados: " + pd.to_numeric(df_viz["Minutos_jugados"], errors="coerce").fillna(0).astype(int).astype(str) + "<br>" +
        f"{x_metric}: " + df_viz[x_metric].apply(lambda v: formatear_valor(x_metric, v)).astype(str) + "<br>" +
        f"{y_metric}: " + df_viz[y_metric].apply(lambda v: formatear_valor(y_metric, v)).astype(str)
    )
    hover_text = [ _sanitize_text(s) for s in hover_series.tolist() ]

    p25_x = np.percentile(df_viz[x_metric], 25)
    p75_x = np.percentile(df_viz[x_metric], 75)
    p25_y = np.percentile(df_viz[y_metric], 25)
    p75_y = np.percentile(df_viz[y_metric], 75)

    return df_viz, hover_text, p25_x, p75_x, p25_y, p75_y


# --- Desempate sólo para coordenadas exactamente coincidentes (separa en un pequeño anillo) ---
def desempatar_coincidentes_exactos(df_viz: pd.DataFrame, x_col: str, y_col: str):
    x = pd.to_numeric(df_viz[x_col], errors="coerce")
    y = pd.to_numeric(df_viz[y_col], errors="coerce")
    x_out = x.copy()
    y_out = y.copy()

    def _min_pos_diff(s: pd.Series) -> float:
        arr = np.sort(s.dropna().unique())
        if arr.size < 2:
            return np.nan
        dif = np.diff(arr)
        dif = dif[dif > 0]
        return float(dif.min()) if dif.size else np.nan

    step_x = _min_pos_diff(x)
    step_y = _min_pos_diff(y)
    span_x = float(x.max() - x.min()) if x.notna().any() else 0.0
    span_y = float(y.max() - y.min()) if y.notna().any() else 0.0

    rx = (step_x * 0.10) if (isinstance(step_x, (float, np.floating)) and np.isfinite(step_x)) else (span_x * 0.005 if span_x > 0 else 1e-6)
    ry = (step_y * 0.10) if (isinstance(step_y, (float, np.floating)) and np.isfinite(step_y)) else (span_y * 0.005 if span_y > 0 else 1e-6)

    grupos = df_viz.groupby([x, y], sort=False).groups
    for _, idx in grupos.items():
        n = len(idx)
        if n > 1:
            ang = np.linspace(0, 2*np.pi, n, endpoint=False)
            x_out.loc[idx] = x.loc[idx].values + rx * np.cos(ang)
            y_out.loc[idx] = y.loc[idx].values + ry * np.sin(ang)

    return x_out, y_out

if len(df_agg) > 0:
    df_viz, hover_text, p25_x, p75_x, p25_y, p75_y = preparar_visualizacion(df_agg, x_metric, y_metric, 0.01, metricas_invertir)
    if df_viz.empty or "Relevancia" not in df_viz.columns:
        st.warning("No se pudo calcular la relevancia de los jugadores. Revisa que las métricas seleccionadas existan en los datos.")
        st.stop()
    # ===== Opciones de visualización =====
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_percentiles = st.checkbox(
            "Mostrar percentiles (P25/P75)",
            value=st.session_state.get("viz_show_percentiles", False),
            key="viz_show_percentiles",
            help="Traza líneas de referencia en los percentiles 25 y 75 de cada métrica seleccionada."
        )
    with col_opt2:
        show_names = st.checkbox(
            "Mostrar nombres de jugadores",
            value=st.session_state.get("viz_show_names", True),
            key="viz_show_names",
            help="Activa las etiquetas con el nombre del jugador; controla cuántas se muestran con la sensibilidad."
        )
    with col_opt3:
        reduce_detail = st.checkbox(
            "Reducir muestra al detalle",
            value=st.session_state.get("viz_reduce_detail", False),
            key="viz_reduce_detail",
            help=(
                "Si está activado, aparecen sliders por métrica para enfocar un rango (zoom). "
                "Si está desactivado, usa un slider para elegir cuántos jugadores mostrar (los más destacados en X+Y)."
            )
        )

    # Sensibilidad sólo cuando se muestran nombres
    if show_names:
        sensibilidad = st.slider(
            "Sensibilidad para mostrar nombres (0 = menos nombres, 100 = más)",
            0, 10, st.session_state.get("viz_sensibilidad", 0), step=1,
            key="viz_sensibilidad",
            help="Determina el umbral de relevancia a partir del cual se muestran etiquetas de nombre."
        )
        umbral = np.percentile(df_viz["Relevancia"], sensibilidad)
        df_viz["Mostrar_nombre"] = (df_viz["Relevancia"] >= umbral)
    else:
        df_viz["Mostrar_nombre"] = False

    # --- Calcular rangos globales para sliders de métricas ---
    min_x, max_x = float(df_viz[x_metric].min()), float(df_viz[x_metric].max())
    min_y, max_y = float(df_viz[y_metric].min()), float(df_viz[y_metric].max())
    if min_x == max_x:
        min_x, max_x = min_x - 1.0, max_x + 1.0
    if min_y == max_y:
        min_y, max_y = min_y - 1.0, max_y + 1.0
    step_x = max((max_x - min_x) / 100, 0.01)
    step_y = max((max_y - min_y) / 100, 0.01)

    # Control del tamaño de muestra o sliders por métrica
    apply_zoom = False
    if reduce_detail:
        with st.expander("Rangos por métrica (zoom)", expanded=True):
            x_def = st.session_state.get("viz_x_range", (min_x, max_x))
            y_def = st.session_state.get("viz_y_range", (min_y, max_y))
            col1, col2 = st.columns(2)
            with col1:
                x_range = st.slider(
                    f"Rango de {x_metric}", min_x, max_x, x_def, step=step_x,
                    key="viz_x_range",
                    help="Límite inferior y superior para la métrica del eje X."
                )
            with col2:
                y_range = st.slider(
                    f"Rango de {y_metric}", min_y, max_y, y_def, step=step_y,
                    key="viz_y_range",
                    help="Límite inferior y superior para la métrica del eje Y."
                )
        df_viz = df_viz[
            (df_viz[x_metric] >= x_range[0]) & (df_viz[x_metric] <= x_range[1]) &
            (df_viz[y_metric] >= y_range[0]) & (df_viz[y_metric] <= y_range[1])
        ]
        apply_zoom = True
    else:
        n_disp = int(len(df_viz))
        top_n_max = int(min(500, max(1, n_disp)))
        if top_n_max <= 1:
            # No mostramos slider si solo hay 1 jugador en muestra
            top_n = 1
            st.caption("Solo 1 jugador cumple los filtros actuales.")
        else:
            min_slider = 1 if n_disp <= 10 else 10
            _default_top = int(st.session_state.get("viz_top_n", top_n_max))
            _default_top = max(min_slider, min(_default_top, top_n_max))
            top_n = st.slider(
                "¿Cuántos jugadores mostrar?",
                min_slider, top_n_max, _default_top, step=1,
                key="viz_top_n",
                help="Muestra los jugadores mejor ubicados hacia la esquina superior derecha (combinación de X e Y)."
            )
        df_viz = df_viz.sort_values("Relevancia", ascending=False).head(top_n)
        x_range = (min_x, max_x)
        y_range = (min_y, max_y)


    # Recalcular percentiles sobre la muestra actual
    p25_x = np.percentile(df_viz[x_metric], 25)
    p75_x = np.percentile(df_viz[x_metric], 75)
    p25_y = np.percentile(df_viz[y_metric], 25)
    p75_y = np.percentile(df_viz[y_metric], 75)

    # Recalcular hover_text en base a la muestra final (evita desalineación con x/y)
    if "Equipo_data_full" in df_viz.columns:
        equipo_series = df_viz["Equipo_data_full"].astype(str)
    else:
        equipo_series = df_viz.get("Equipo_data", pd.Series("", index=df_viz.index)).astype(str)
    hover_series = (
        "Jugador: " + df_viz["Nombre_transfermarket"].astype(str) + "<br>" +
        "Equipo: " + equipo_series.astype(str) + "<br>" +
        "Minutos jugados: " + pd.to_numeric(df_viz["Minutos_jugados"], errors="coerce").fillna(0).astype(int).astype(str) + "<br>" +
        f"{x_metric}: " + df_viz[x_metric].apply(lambda v: formatear_valor(x_metric, v)).astype(str) + "<br>" +
        f"{y_metric}: " + df_viz[y_metric].apply(lambda v: formatear_valor(y_metric, v)).astype(str)
    )
    hover_text = [ _sanitize_text(s) for s in hover_series.tolist() ]

    # --- Subtítulo dinámico centrado ---
    pos_diminutivos = [diminutivos_pos.get(p, p[:2].upper()) for p in pos_det_sel]
    texto_posiciones = ", ".join(sorted(set(pos_diminutivos)))
    texto_torneos = ", ".join(sorted(set(torneos_sel)))
    subtitulo = f"{texto_posiciones} de {texto_torneos} en {temporada}"

    with st.spinner("Cargando gráfico simple..."):
        # --- Customdata JSON-safe para hovertemplate ---
        nombres_cd = df_viz["Nombre_transfermarket"].astype(str).map(_sanitize_text)
        equipos_cd = equipo_series.astype(str).map(_sanitize_text)
        minutos_cd = pd.to_numeric(df_viz["Minutos_jugados"], errors="coerce").fillna(0).astype(int)
        x_fmt_cd = df_viz[x_metric].apply(lambda v: formatear_valor(x_metric, v)).astype(str).map(_sanitize_text)
        y_fmt_cd = df_viz[y_metric].apply(lambda v: formatear_valor(y_metric, v)).astype(str).map(_sanitize_text)
        customdata = np.column_stack([
            nombres_cd.values,
            equipos_cd.values,
            minutos_cd.values,
            x_fmt_cd.values,
            y_fmt_cd.values,
        ]).tolist()
        # Colores por equipo usando 'Color primario' si existe; si es nulo, asignar un color único
        palette = (
            px.colors.qualitative.Dark24 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3 +
            px.colors.qualitative.Pastel1 + px.colors.qualitative.Safe
        )
        equipos_series = df_viz["Equipo_data"].fillna("N/A").astype(str) if "Equipo_data" in df_viz.columns else pd.Series(["N/A"]*len(df_viz))
        prim_col_name = "Color primario"
        if prim_col_name in df_viz.columns:
            prim_series = df_viz[prim_col_name].astype(object)
        else:
            prim_series = pd.Series([None] * len(df_viz))

        def _norm_hex(c):
            return _clean_hex_color(c)

        team_color = {}
        used_colors = set()
        # Asignar colores primarios válidos
        for team, col in zip(equipos_series, prim_series):
            hexcol = _norm_hex(col)
            if hexcol and team not in team_color:
                team_color[team] = hexcol
                used_colors.add(hexcol.upper())

        # Fallback: colores únicos para equipos sin color primario válido
        palette_iter = iter([c for c in palette if c.upper() not in used_colors])
        for team in equipos_series.unique():
            if team not in team_color:
                try:
                    nxt = next(palette_iter)
                except StopIteration:
                    # Generar un color pseudo-único basado en hash del equipo
                    nxt = "#%06X" % (abs(hash(team)) & 0xFFFFFF)
                    # Evitar colisión con existentes
                    if nxt.upper() in used_colors:
                        nxt = "#%06X" % ((abs(hash(team + "_alt")) & 0xFFFFFF))
                team_color[team] = nxt.upper()
                used_colors.add(team_color[team])

        # Seguridad: si por alguna razón hay duplicados en fallback, reasignar hasta que sean únicos
        seen = set()
        for t in list(team_color.keys()):
            c = team_color[t].upper()
            if c in seen and (t not in equipos_series.unique() or _norm_hex(df_viz.loc[equipos_series == t, prim_col_name].iloc[0] if prim_col_name in df_viz.columns else None) is None):
                # duplicado en fallback: buscar otro color libre
                for cand in palette:
                    if cand.upper() not in used_colors:
                        team_color[t] = cand.upper()
                        used_colors.add(cand.upper())
                        break
            seen.add(team_color[t].upper())

        color_values = [team_color.get(t, "#D3D3D3") for t in equipos_series]
        color_values = [cv if isinstance(cv, str) and _hex_re.match(cv) else "#D3D3D3" for cv in color_values]

        # Separar únicamente puntos con coordenadas (x,y) exactamente iguales
        x_plot, y_plot = desempatar_coincidentes_exactos(df_viz, x_metric, y_metric)

        # Crear gráfico de burbujas
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers+text",
            text=[ _sanitize_text(x) for x in np.where(df_viz["Mostrar_nombre"], df_viz["Nombre_transfermarket"].astype(str), "") ],
            textposition="top center",
            marker=dict(
                size=12,
                color=color_values,
                line=dict(width=1, color='black'),
                opacity=1
            ),
            customdata=customdata,
            meta=dict(x_label=_sanitize_text(x_metric), y_label=_sanitize_text(y_metric)),
            hovertemplate=(
                "Jugador: %{customdata[0]}<br>"
                "Equipo: %{customdata[1]}<br>"
                "Minutos jugados: %{customdata[2]}<br>"
                "%{meta.x_label}: %{customdata[3]}<br>"
                "%{meta.y_label}: %{customdata[4]}<extra></extra>"
            ),
            textfont=dict(color=ejes_color)
        ))
        # Líneas verticales y horizontales de percentiles (condicional)
        if show_percentiles:
            fig.add_vline(x=p25_x, line_dash="dot", line_color=linea_25)
            fig.add_vline(x=p75_x, line_dash="dot", line_color=linea_75)
            fig.add_hline(y=p25_y, line_dash="dot", line_color=linea_25)
            fig.add_hline(y=p75_y, line_dash="dot", line_color=linea_75)
        # Mejorar visualización de ejes y ocultar cuadrícula, con margen superior para evitar choques con el subtítulo
        if apply_zoom:
            fig.update_xaxes(range=[x_range[0], x_range[1]], showgrid=False, zeroline=False)
            y_span = max(y_range[1] - y_range[0], 1e-9)
            y_pad = max(0.05 * y_span, 0.01)  # ~5% de margen superior
            fig.update_yaxes(range=[y_range[0], y_range[1] + y_pad], showgrid=False, zeroline=False, automargin=True)
        else:
            fig.update_xaxes(showgrid=False, zeroline=False)
            y_span = max_y - min_y
            if y_span <= 0:
                y_span = 1.0
            y_pad = max(0.05 * y_span, 0.01)
            fig.update_yaxes(range=[min_y, max_y + y_pad], showgrid=False, zeroline=False, automargin=True)

        modo_label = "por 90’" if modo_90 else "totales"
        # Línea 1 del subtítulo: inteligencia por posiciones (GK, DEF, MED, DEL)
        pos_line = construir_linea_posiciones(df_filtros, pos_gen_sel, pos_det_sel)
        # Línea 2 del subtítulo: torneos y países
        line2 = f"{torneos_text} · Equipos de {paises_text}"
        titulo_fig = f"{x_metric} vs {y_metric} — {temporada} ({modo_label})"
        # Compactar subtítulo con spans y line-height bajo (no bold, más compacto)
        subtitulo_fig = (
            f"<span style='display:block; font-size:12px; font-weight:400; line-height:1.0'>{_sanitize_text(pos_line)}</span>"
            f"<span style='display:block; font-size:12px; font-weight:400; line-height:1.0; margin-top:-6px'>{_sanitize_text(line2)}</span>"
        )

        fig.update_layout(
            height=500,
            margin=dict(t=70, l=40, r=40, b=40),  # margen superior reducido
            showlegend=False,
            xaxis=dict(
                title=dict(
                    text=x_metric,
                    font=dict(color=ejes_color, size=12)
                ),
                tickformat=".0%" if x_metric in metricas_porcentaje else None,
                tickfont=dict(color=ejes_color),
                color=ejes_color
            ),
            yaxis=dict(
                title=dict(
                    text=y_metric,
                    font=dict(color=ejes_color, size=12)
                ),
                tickformat=".0%" if y_metric in metricas_porcentaje else None,
                tickfont=dict(color=ejes_color),
                color=ejes_color
            ),
            title=dict(
                text=f"{titulo_fig}<br>{subtitulo_fig}",
                x=0.5,
                xanchor="center",
                font=dict(size=16, color=font_color)
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
        )
        st.plotly_chart(fig, use_container_width=True)
        # Pie de nota: solo muestra y filtros que no están en el subtítulo
        n_jug = len(df_viz)
        n_eq = (
            df_viz["Equipo_data"].nunique() if "Equipo_data" in df_viz.columns
            else (len(set(equipos_sel)) if equipos_sel else 0)
        )
        pie_1 = (
            f"Muestra: {n_jug} jugadores, {n_eq} equipos · "
            f"Min≥{min_jugados}, M90s≥{min_m90s}, Edad {int(edad_range[0])}-{int(edad_range[1])} · "
            f"Nac: {nac_text if nac_text!='' else 'N/A'}"
        )
        pie_2 = "Elaborado por Jurgen Schmidt"
        st.caption(pie_1 + "  \n" + pie_2)

        if show_percentiles:
            st.info("ℹ️ La línea verde representa el percentil 75 y la línea roja el percentil 25 de cada métrica seleccionada.")


st.caption(f"📅 Última actualización del dataset: **{fecha_actualizacion}**")