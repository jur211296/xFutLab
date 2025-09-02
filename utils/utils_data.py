import pandas as pd
import streamlit as st

@st.cache_data
def cargar_datos(ruta_parquet: str):
    """
    Carga EXCLUSIVA desde Parquet "gold" (dataset ya enriquecido).
    - Requiere una ruta .parquet válida.
    - No hay soporte para CSV ni limpiezas adicionales aquí.
    """
    import os
    if not isinstance(ruta_parquet, str) or not ruta_parquet.lower().endswith(".parquet"):
        raise ValueError("Se esperaba una ruta .parquet 'gold'.")
    if not os.path.exists(ruta_parquet):
        raise FileNotFoundError(f"No existe el archivo Parquet: {ruta_parquet}")
    return pd.read_parquet(ruta_parquet)

def cargar_metricas(ruta_metricas):
    return pd.read_excel(ruta_metricas)


# ===============================
# === MÉTRICAS PERSONALIZADAS ===
# ===============================

def calcular_diferenciales(df):
    """Calcula métricas diferenciales como Goles-xG, Asistencias-xA, etc."""
    if "Asistencias" in df.columns and "xAsistencias" in df.columns:
        df["Asistencias-xA"] = (df["Asistencias"] - df["xAsistencias"]).round(1)
    if "Goles" in df.columns and "xGoles" in df.columns:
        df["Goles-xG"] = (df["Goles"] - df["xGoles"]).round(1)
    if "Goles_recibidos" in df.columns and "xGoles_recibidos" in df.columns:
        df["Goles-xG (en contra)"] = (df["Goles_recibidos"] - df["xGoles_recibidos"]).round(1)
    return df

def calcular_porcentajes(df):
    """Calcula métricas de porcentaje a partir de columnas base."""
    formulas = {
        "Tiros_libres_directos_exitosos_perc":       ("Tiros_libres_directos_exitosos", "Tiros_libres_directos"),
        "Penaltis_favor_exitosos_perc":              ("Penaltis_favor_exitosos", "Penaltis_favor"),
        "Duelos_ganados_perc":                       ("Duelos_ganados", "Duelos"),
        "Duelos_defensivos_ganados_perc":            ("Duelos_defensivos_ganados", "Duelos_defensivos"),
        "Duelos_aereos_ganados_perc":                ("Duelos_aereos_ganados", "Duelos_aereos"),
        "Centros_exitosos_perc":                     ("Centros_exitosos", "Centros"),
        "Centros_izquierda_exitosos_perc":           ("Centros_izquierda_exitosos", "Centros_izquierda"),
        "Centros_derecha_exitosos_perc":             ("Centros_derecha_exitosos", "Centros_derecha"),
        "Regates_exitosos_perc":                     ("Regates_exitosos", "Regates"),
        "Pases_exitosos_perc":                       ("Pases_exitosos", "Pases"),
        "Pases_hacia_adelante_exitosos_perc":        ("Pases_hacia_adelante_exitosos", "Pases_hacia_adelante"),
        "Pases_hacia_atras_exitosos_perc":           ("Pases_hacia_atras_exitosos", "Pases_hacia_atras"),
        "Pases_laterales_exitosos_perc":             ("Pases_laterales_exitosos", "Pases_laterales"),
        "Pases_cortos_medios_exitosos_perc":         ("Pases_cortos_medios_exitosos", "Pases_cortos_medios"),
        "Pases_largos_exitosos_perc":                ("Pases_largos_exitosos", "Pases_largos"),
        "Desmarques_exitosos_perc":                  ("Desmarques_exitosos", "Desmarques"),
        "Pases_ultimo_tercio_exitosos_perc":         ("Pases_ultimo_tercio_exitosos", "Pases_ultimo_tercio"),
        "Pases_area_penalti_exitosos_perc":          ("Pases_area_penalti_exitosos", "Pases_area_penalti"),
        "Pases_profundidad_exitosos_perc":           ("Pases_profundidad_exitosos", "Pases_profundidad"),
        "Pases_progresivos_exitosos_perc":           ("Pases_progresivos_exitosos", "Pases_progresivos"),
        "Duelos_ofensivos_ganados_perc":             ("Duelos_ofensivos_ganados", "Duelos_ofensivos"),
        "Disparos_puerta_perc":                      ("Disparos_puerta", "Disparos"),
        "Goles_conversion_disparos_perc":            ("Goles", "Disparos"),
        "Goles_conversion_disparosPuerta_perc":      ("Goles", "Disparos_puerta"),
        "Paradas_perc":                              ("Paradas", "Disparos_recibidos"),
    }
    for nueva, (num, den) in formulas.items():
        if num in df.columns and den in df.columns:
            df[nueva] = df[num] / df[den]
            df[nueva] = df[nueva].replace([float('inf'), -float('inf')], None)
            df[nueva] = df[nueva].clip(upper=1).fillna(0)
    return df

def calcular_metricas_por_90(df, df_metricas):
    """Crea columnas /90 para métricas que lo requieren según df_metricas."""
    for _, row in df_metricas[df_metricas["Considerar"] == "/90"].iterrows():
        base_col = row["Nombre en csv"]
        col_90 = base_col + "/90"
        if base_col in df.columns and col_90 not in df.columns:
            df[col_90] = (pd.to_numeric(df[base_col], errors="coerce") / df["M90s_jugados"]).round(1)
    return df

def renombrar_columnas(df, df_metricas):
    """Renombra columnas del dataframe para visualización según df_metricas."""
    renombrar_cols = {}
    for _, row in df_metricas.iterrows():
        considerar = row["Considerar"]
        csv = row["Nombre en csv"]
        app = row["Nombre en app"]
        if considerar == "/90":
            renombrar_cols[csv + "/90"] = app
        elif considerar in ["Totales", "Porcentaje"]:
            renombrar_cols[csv] = app
    df = df.rename(columns=renombrar_cols)
    return df

def construir_diccionarios(df_metricas):
    """Construye diccionarios de configuración y lista de métricas de porcentaje según df_metricas."""
    considerar_dict = {}
    tipos_dict = {}
    metricas_porcentaje = []
    for _, fila in df_metricas.iterrows():
        if fila["Considerar"] != "NO":
            considerar_dict[fila["Nombre en app"]] = fila["Considerar"]
            tipos_dict[fila["Nombre en app"]] = fila["Tipo"]
            if fila["Considerar"] == "Porcentaje":
                metricas_porcentaje.append(fila["Nombre en app"])
    return considerar_dict, tipos_dict, metricas_porcentaje

def aplicar_metricas_personalizadas(df, df_metricas):
    """
    Aplica el cálculo de métricas personalizadas:
    - Diferenciales (Goles-xG, Asistencias-xA)
    - Porcentajes a partir de columnas base
    - Métricas por 90 minutos
    - Renombrado de columnas para visualización
    - Diccionarios de configuración
    """
    df = calcular_diferenciales(df)
    df = calcular_porcentajes(df)
    df = calcular_metricas_por_90(df, df_metricas)
    df = renombrar_columnas(df, df_metricas)
    considerar_dict, tipos_dict, metricas_porcentaje = construir_diccionarios(df_metricas)
    # Métricas defensivas a invertir
    metricas_invertir = [
        "Goles en contra", "Goles esperados en contra", "Disparos en contra", "Goles - xG (en contra)",
        "Goles en contra /90", "Goles esperados en contra /90", "Disparos en contra /90", "Goles - xG (en contra) /90"
    ]
    return df, considerar_dict, tipos_dict, metricas_porcentaje, metricas_invertir

# ===============================
# === CONFIGURACIÓN GENERAL ===
# ===============================

# === TIPOS Y MÉTRICAS POR POSICIÓN ===

tipos_default_por_posicion = {
    "Portero": ["Portero", "Construcción"],
    "Defensa central": ["Físicas", "Construcción"],
    "Lateral derecho": ["Físicas", "Construcción", "Ofensivas"],
    "Lateral izquierdo": ["Físicas", "Construcción", "Ofensivas"],
    "Pivote": ["Físicas", "Construcción"],
    "Mediocentro": ["Ofensivas", "Construcción", "Físicas"],
    "Interior derecho": ["Ofensivas", "Construcción", "Físicas"],
    "Interior izquierdo": ["Ofensivas", "Construcción", "Físicas"],
    "Mediocentro ofensivo": ["Ofensivas", "Construcción"],
    "Extremo derecho": ["Ofensivas", "Construcción", "Físicas"],
    "Extremo izquierdo": ["Ofensivas", "Construcción", "Físicas"],
    "Delantero centro": ["Ofensivas", "Físicas", "Construcción"],
}

metricas_default_por_posicion = {
    "Portero": ["Goles - xG (en contra)", "Goles - xG (en contra) /90",
                        "Goles en contra", "Goles en contra /90",
                        "Goles en contra evitados", "Goles en contra evitados /90",
                        "Disparos en contra", "Disparos en contra /90",
                        "Paradas", "Paradas /90",
                        "Paradas %",
                        "Salidas", "Salidas /90",
                        "Duelos aéreos de portero", "Duelos aéreos de portero /90",
                        "Pases largos exitosos %",
                        "Pases cortos/medios exitosos %"],
    "Defensa central": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Duelos aéreos", "Duelos aéreos /90",
                        "Duelos aéreos ganados %",
                        "Intercepciones", "Intercepciones /90",
                        "Pases largos", "Pases largos /90",
                        "Pases largos exitosos %",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases progresivos exitosos %"],
    "Lateral derecho": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Centros desde la derecha", "Centros desde la derecha /90",
                        "Centros desde la derecha exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates exitosos %",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases al último tercio", "Pases al último tercio /90"],
    "Carrilero derecho": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Centros desde la derecha", "Centros desde la derecha /90",
                        "Centros desde la derecha exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates exitosos %",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases al último tercio", "Pases al último tercio /90"],
    "Lateral izquierdo": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Centros desde la izquierda", "Centros desde la izquierda /90",
                        "Centros desde la izquierda exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates exitosos %",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases al último tercio", "Pases al último tercio /90"],
    "Carrilero izquierdo": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Centros desde la izquierda", "Centros desde la izquierda /90",
                        "Centros desde la izquierda exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates exitosos %",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases al último tercio", "Pases al último tercio /90"],
    "Pivote": ["Acciones defensivas ganadas", "Acciones defensivas ganadas /90",
                        "Duelos defensivos", "Duelos defensivos /90",
                        "Duelos defensivos ganados %",
                        "Duelos aéreos ganados %",
                        "Intercepciones", "Intercepciones /90",
                        "Pases progresivos", "Pases progresivos /90",
                        "Pases progresivos exitosos %",
                        "Pases cortos/medios exitosos %",
                        "Pases largos", "Pases largos /90",
                        "Pases largos exitosos %"],
    "Interior derecho": ["Duelos defensivos ganados %",
                         "Asistencias a disparos", "Asistencias a disparos /90",
                         "Jugadas clave", "Jugadas clave /90",
                         "Pases al área penalti", "Pases al área penalti /90",
                         "Pases al área penalti exitosos %",
                         "Pases al último tercio", "Pases al último tercio /90",
                         "Pases al último tercio exitosos %",
                         "Pases en profundidad", "Pases en profundidad /90",
                         "Pases progresivos", "Pases progresivos /90",
                         "Pases progresivos exitosos %"],
    "Mediocentro": ["Duelos defensivos ganados %",
                         "Asistencias a disparos", "Asistencias a disparos /90",
                         "Jugadas clave", "Jugadas clave /90",
                         "Pases al área penalti", "Pases al área penalti /90",
                         "Pases al área penalti exitosos %",
                         "Pases al último tercio", "Pases al último tercio /90",
                         "Pases al último tercio exitosos %",
                         "Pases en profundidad", "Pases en profundidad /90",
                         "Pases progresivos", "Pases progresivos /90",
                         "Pases progresivos exitosos %"],
    "Interior izquierdo": ["Duelos defensivos ganados %",
                         "Asistencias a disparos", "Asistencias a disparos /90",
                         "Jugadas clave", "Jugadas clave /90",
                         "Pases al área penalti", "Pases al área penalti /90",
                         "Pases al área penalti exitosos %",
                         "Pases al último tercio", "Pases al último tercio /90",
                         "Pases al último tercio exitosos %",
                         "Pases en profundidad", "Pases en profundidad /90",
                         "Pases progresivos", "Pases progresivos /90",
                         "Pases progresivos exitosos %"],
    "Mediocentro ofensivo": ["Asistencias a disparos", "Asistencias a disparos /90",
                         "Jugadas clave", "Jugadas clave /90",
                         "Pases al área penalti", "Pases al área penalti /90",
                         "Pases al área penalti exitosos %",
                         "Pases al último tercio", "Pases al último tercio /90",
                         "Pases al último tercio exitosos %",
                         "Asistencias esperadas", "Asistencias esperadas /90",
                         "Acciones de ataque exitosas", "Acciones de ataque exitosas /90",
                         "Disparos a puerta", "Disparos a puerta /90",
                         "Regates exitosos %"],
    "Extremo derecho": ["Centros desde la derecha", "Centros desde la derecha /90",
                        "Centros desde la derecha exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates", "Regates /90",
                        "Regates exitosos %",
                        "Asistencias esperadas", "Asistencias esperadas /90",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Goles - xG", "Goles - xG /90",
                        "Disparos a puerta", "Disparos a puerta /90",
                        "Toques en el área penalti", "Toques en el área penalti /90"],
    "Extremo izquierdo": ["Centros desde la izquierda", "Centros desde la izquierda /90",
                        "Centros desde la izquierda exitosos %",
                        "Centros desde último tercio", "Centros desde último tercio /90",
                        "Regates", "Regates /90",
                        "Regates exitosos %",
                        "Asistencias esperadas", "Asistencias esperadas /90",
                        "Asistencias a disparos", "Asistencias a disparos /90",
                        "Goles - xG", "Goles - xG /90",
                        "Disparos a puerta", "Disparos a puerta /90",
                        "Toques en el área penalti", "Toques en el área penalti /90"],
    "Delantero centro": ["Disparos a puerta", "Disparos a puerta /90",
                         "Disparos a puerta %",
                         "Goles - xG", "Goles - xG /90",
                         "Goles esperados", "Goles esperados /90",
                         "Goles por disparo a puerta %",
                         "Goles de cabeza", "Goles de cabeza /90",
                         "Goles no penales", "Goles no penales /90",
                         "Asistencias a disparos", "Asistencias a disparos /90",
                         "Asistencias esperadas", "Asistencias esperadas /90",
                         "Toques en el área penalti", "Toques en el área penalti /90"],
}

# Diccionario de diminutivos para posiciones detalladas
diminutivos_pos = {
    "Portero": "PT",
    "Defensa central": "DFC",
    "Lateral derecho": "LD",
    "Lateral izquierdo": "LI",
    "Pivote": "MCD",
    "Mediocentro": "MC",
    "Interior derecho": "MC",
    "Interior izquierdo": "MC",
    "Mediocentro ofensivo": "MCO",
    "Mediapunta": "MP",
    "Extremo derecho": "ED",
    "Extremo izquierdo": "EI",
    "Delantero centro": "DC"
}

colores_paises = {
        "Peru": "#DB5375",
        "Argentina": "#92DCE5",
        "Brasil": "#9BE564",
        "Chile": "#e63946",
        "Colombia": "#D7F75B",
        "Ecuador": "#D19C1D",
        "Uruguay": "#467599",
        "Bolivia": "#9CFC97",
        "Venezuela": "#AA767C",
        "Paraguay": "#232ED1"
    }



# ===============================
# 📊 OBTENER PERCENTILES
# ===============================
import numpy as np
from scipy.stats import rankdata

# Será completado al aplicar_metricas_personalizadas
metricas_porcentaje = []

def obtener_percentiles(jugador, df_base, metricas, metricas_invertir=None):
    """
    Calcula percentiles (0–100) del jugador respecto a `df_base` para las métricas indicadas.
    Soporta `jugador` como Series, dict o DataFrame de 1 fila.

    Reglas:
    - Percentil = proporción de valores de la muestra <= valor del jugador (ECDF * 100).
    - Si la métrica está en `metricas_invertir`, se refleja: 100 - percentil.
    - No se reemplazan 0→NaN en métricas diferenciales (Goles-xG, Asistencias-xA, y sus /90, con o sin espacios/hífen).
    - En el resto de métricas, 0 se trata como missing para evitar sesgos por ceros estructurales.
    """
    # Normalizar jugador a una "fila" accesible
    if isinstance(jugador, pd.DataFrame):
        jrow = jugador.iloc[0] if len(jugador) > 0 else pd.Series(dtype=float)
    else:
        jrow = jugador

    # Conjunto de métricas donde 0 es un valor válido (no convertir a NaN)
    bases_diff = [
        "Goles - xG", "Asistencias - xA", "Goles - xG (en contra)",
    ]

    def _canon(name: str) -> str:
        return str(name).replace(" ", "").replace("_", "").replace("-", "").lower()

    diffs_canon = set()
    for b in bases_diff:
        diffs_canon.add(_canon(b))
        diffs_canon.add(_canon(f"{b} /90"))

    inv_set = set(metricas_invertir) if metricas_invertir is not None else set()

    def _get_scalar(row_like, key):
        """Obtiene un escalar seguro desde row_like[key]."""
        try:
            val = row_like[key] if isinstance(row_like, (pd.Series, dict)) else getattr(row_like, key)
        except Exception:
            try:
                val = row_like.get(key, None)
            except Exception:
                val = None
        # Si es secuencia o Serie, tomar el primer elemento
        if isinstance(val, (pd.Series, pd.Index, list, tuple)):
            if len(val) == 0:
                return None
            try:
                return val.iloc[0] if hasattr(val, "iloc") else val[0]
            except Exception:
                return None
        return val

    out = []

    for m in metricas:
        # Valor del jugador
        raw_val = _get_scalar(jrow, m)
        try:
            v = float(raw_val)
        except Exception:
            v = float("nan")

        # Serie base limpia
        if m not in df_base.columns:
            out.append(0)
            continue
        serie = pd.to_numeric(df_base[m], errors="coerce").replace([float('inf'), -float('inf')], pd.NA)

        # 0 como NaN salvo en diferenciales
        if _canon(m) not in diffs_canon:
            serie = serie.replace(0, pd.NA)
        serie = serie.dropna()

        if len(serie) == 0 or pd.isna(v):
            pct = 0.0
        else:
            # ECDF inclusiva
            pct = float((serie <= v).mean()) * 100.0
            if m in inv_set:
                pct = 100.0 - pct
        # Clipping y entero
        pct = max(0.0, min(100.0, pct))
        out.append(int(round(pct)))

    return out

# Función de formateo general para tablas, tooltips, hovertext
def formatear_valor(metrica, valor):
    try:
        if metrica in metricas_porcentaje:
            if pd.notnull(valor) and isinstance(valor, (int, float)):
                return f"{valor*100 :.0f}%"
            else:
                return ""
        elif isinstance(valor, float):
            return f"{valor:.1f}"
        elif isinstance(valor, int):
            return str(valor)
        elif pd.isnull(valor):
            return ""
        else:
            return str(valor)
    except Exception:
        return str(valor)
    
metricas_fisicas = [
    "Acciones defensivas ganadas", "Duelos defensivos", "Duelos defensivos ganados", "Duelos defensivos ganados %",
    "Duelos aéreos", "Duelos aéreos ganados", "Duelos aéreos ganados %",
    "Entradas agresivas", "Intercepciones", "Faltas realizadas",
    "Duelos ofensivos", "Duelos ofensivos ganados", "Duelos ofensivos ganados %",
    "Acciones defensivas ganadas /90", "Duelos defensivos /90", "Duelos defensivos ganados /90",
    "Duelos aéreos /90", "Duelos aéreos ganados /90", "Entradas agresivas /90", "Intercepciones /90",
    "Faltas realizadas /90", "Duelos ofensivos /90", "Duelos ofensivos ganados /90"
]

metricas_centros = [
    "Centros totales /90", "Centros totales exitosos /90", "Centros desde la izquierda /90",
    "Centros desde la izquierda exitosos /90", "Centros desde la derecha /90", "Centros desde la derecha exitosos /90",
    "Centros al área pequeña /90", "Centros desde último tercio /90", "Centros totales exitosos %",
    "Centros desde la izquierda exitosos %", "Centros desde la derecha exitosos %",
    "Centros totales", "Centros totales exitosos", "Centros desde la izquierda", "Centros desde la izquierda exitosos",
    "Centros desde la derecha", "Centros desde la derecha exitosos", "Centros al área pequeña",
    "Centros desde último tercio"
]

metricas_construccion_general = [
    "Pases recibidos /90", "Pases largos recibidos /90", "Pases totales /90", "Pases totales exitosos /90",
    "Pases hacia adelante /90", "Pases hacia atrás /90", "Pases cortos/medios /90", "Pases cortos/medios exitosos /90",
    "Pases largos /90", "Pases largos exitosos /90", "Pases progresivos /90", "Pases progresivos exitosos /90",
    "Pases totales exitosos %", "Pases cortos/medios exitosos %", "Pases largos exitosos %", "Pases progresivos exitosos %",
    "Pases recibidos", "Pases largos recibidos", "Pases totales", "Pases totales exitosos",
    "Pases hacia adelante", "Pases hacia atrás", "Pases cortos/medios", "Pases cortos/medios exitosos",
    "Pases largos", "Pases largos exitosos", "Pases progresivos", "Pases progresivos exitosos"
]

metricas_construccion_ofensiva = [
    "Toques en el área penalti /90", "Faltas recibidas /90", "Desmarques /90", "Desmarques exitosos /90",
    "Jugadas clave /90", "Pases al último tercio /90", "Pases al último tercio exitosos /90",
    "Pases al área penalti exitosos /90", "Pases en profundidad exitosos /90", "Asistencias /90",
    "Asistencias esperadas /90", "Asistencias a disparos /90", "Asistencias - xA /90",
    "Desmarques exitosos %", "Pases al último tercio exitosos %",
    "Toques en el área penalti", "Faltas recibidas", "Desmarques", "Desmarques exitosos", "Jugadas clave",
    "Pases al último tercio", "Pases al último tercio exitosos", "Pases al área penalti exitosos",
    "Pases en profundidad exitosos", "Asistencias", "Asistencias esperadas", "Asistencias a disparos", "Asistencias - xA"
]

metricas_ofensivas = [
    "Regates /90", "Regates exitosos /90", "Ataques en profundidad /90", "Goles /90", "Goles no penales /90",
    "Goles esperados /90", "Goles de cabeza /90", "Disparos /90", "Disparos a puerta /90",
    "Acciones de ataque exitosas /90", "Goles - xG /90", "Regates exitosos %", "Disparos a puerta %",
    "Goles por disparo %", "Goles por disparo a puerta %",
    "Regates", "Regates exitosos", "Ataques en profundidad", "Goles", "Goles no penales",
    "Goles esperados", "Goles de cabeza", "Disparos", "Disparos a puerta",
    "Acciones de ataque exitosas", "Goles - xG"
]

metricas_balon_parado = [
    "Tiros libres /90", "Tiros libres directos /90", "Tiros libres directos exitosos /90",
    "Tiros de esquina /90", "Penaltis lanzados /90", "Penaltis lanzados exitosos /90",
    "Tiros libres directos exitosos %", "Penaltis lanzados exitosos %",
    "Tiros libres", "Tiros libres directos", "Tiros libres directos exitosos",
    "Tiros de esquina", "Penaltis lanzados", "Penaltis lanzados exitosos"
]

metricas_portero = [
    "Goles en contra",
    "Goles esperados en contra",
    "Porterías imbatidas",
    "Disparos en contra",
    "Goles en contra evitados",
    "Paradas",
    "Paradas %",
    "Salidas",
    "Duelos aéreos de portero",
    "Goles - xG (en contra)",
    "Goles en contra /90",
    "Goles esperados en contra /90",
    "Porterías imbatidas /90",
    "Disparos en contra /90",
    "Goles en contra evitados /90",
    "Paradas /90",
    "Salidas /90",
    "Duelos aéreos de portero /90",
    "Goles - xG (en contra) /90"
]