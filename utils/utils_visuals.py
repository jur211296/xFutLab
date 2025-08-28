# visuals.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from mplsoccer import Radar, grid
from utils.utils_data import obtener_percentiles, formatear_valor
from mplsoccer import PyPizza
import textwrap
import streamlit as st

DEFAULT_COLORES_RADAR = ['#238b23', '#ff3909', '#00c1ff']

# Funciones privadas auxiliares

def _dividir_metrica(m):
    palabras = m.split()
    lineas = []
    linea_actual = ""
    for palabra in palabras:
        if len(linea_actual + " " + palabra) <= 15:
            linea_actual = (linea_actual + " " + palabra).strip()
        else:
            lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)
    return "<br>".join(lineas)

def _invertir_metricas(df, metricas, invertir):
    df_copy = df.copy()
    for m in metricas:
        if m in invertir:
            df_copy[m] = -df_copy[m]
    return df_copy

# Helper seguro para detectar el tema aunque la versión de Streamlit no sea 1.46+
def _get_theme_type():
    ctx = getattr(st, "context", None)
    if ctx is not None and hasattr(ctx, "theme"):
        theme = getattr(ctx, "theme", None)
        if theme is not None:
            t = getattr(theme, "type", None)
            if t in ("light", "dark"):
                return t
    # Intento 2: inferir por el color de texto configurado
    try:
        txt = st.get_option("theme.textColor")
    except Exception:
        txt = None
    if isinstance(txt, str) and txt.startswith("#"):
        # Normalizar #RGB a #RRGGBB si aplica
        hexv = txt
        if len(hexv) == 4:
            hexv = "#" + "".join([c * 2 for c in hexv[1:]])
        try:
            r = int(hexv[1:3], 16) / 255.0
            g = int(hexv[3:5], 16) / 255.0
            b = int(hexv[5:7], 16) / 255.0
            # Luminancia relativa del color de texto; si el texto es muy claro -> tema oscuro
            luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "dark" if luminancia > 0.7 else "light"
        except Exception:
            pass
    # Intento 3: opción de configuración (themes en config.toml)
    base = st.get_option("theme.base")
    if base in ("light", "dark"):
        return base
    # Fallback
    return "light"

# ===============================
# 🛡️ RADAR DE PERCENTILES (PLOTLY, estilo limpio)
# ===============================
@st.cache_data
def crear_radar_percentil_plotly(jugadores, etiquetas, seleccionadas, df_comparacion, colores=None, titulo=None, subtitulo=None, modo_claro=None):
    # Autodetección del tema si no se especifica
    if modo_claro is None:
        try:
            theme_type = _get_theme_type()
        except Exception:
            theme_type = "light"
        modo_claro = (theme_type == "light")
    seleccionadas = sorted(seleccionadas)
    categorias = seleccionadas

    percentiles_data = []
    valores_data = []
    for jugador in jugadores:
        p = obtener_percentiles(jugador, df_comparacion, seleccionadas)
        percentiles_data.append(p + [p[0]])
        valores_data.append([jugador[m] for m in seleccionadas] + [jugador[seleccionadas[0]]])
    # En lugar de transponer directamente, organizamos por punto
    customdata = [[(valores_data[j][i], percentiles_data[j][i]) for j in range(len(jugadores))] for i in range(len(seleccionadas))]

    if colores is None:
        colores = [
            "#ee247c",  # fucsia (Leão)
            "#278cd5",  # celeste (Rodrygo)
            "#44ab69",  # verde (Nico Williams)
        ]
        colores_borde = [
            "#ee247c",   # fucsia
            "#278cd5",   # celeste
            "#44ab69",      # verde
        ]
    else:
        colores_borde = colores

    fig = go.Figure()
    for i in reversed(range(len(jugadores))):
        hover_lines = [f"{etiquetas[j]}: %{{customdata[{j}][0]}} (%{{customdata[{j}][1]}})" for j in range(len(etiquetas))]
        hovertemplate = "<b>%{theta}</b><br>" + "<br>".join(hover_lines) + "<extra></extra>"
        fig.add_trace(go.Scatterpolar(
            r=percentiles_data[i],
            theta=categorias,
            fill='toself',
            name=etiquetas[i],
            line_color=colores_borde[i % len(colores_borde)],
            line=dict(width=4),  # Línea más gruesa
            marker=dict(size=12, color=colores_borde[i % len(colores_borde)]),
            opacity=1,
            mode="lines+markers",
            hoverinfo="skip",  # Para estilo limpio, puedes poner "text" si quieres hover
            customdata=customdata,
            hovertemplate=hovertemplate
        ))

    paper_bgcolor = 'rgba(0,0,0,0)'
    plot_bgcolor = 'rgba(0,0,0,0)'
    grid_color = 'rgba(0,0,0,0.12)' if modo_claro else 'rgba(255,255,255,0.12)'
    font_color = "#000000" if modo_claro else "white"
    annotation_color = "#000000" if modo_claro else "white"
    legend_font_color = "#000000" if modo_claro else "white"

    fig.update_layout(
        autosize=True, width=600, height=600,
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                showline=False,
                ticks='',
                gridcolor=grid_color,
                gridwidth=1.5,
                range=[0, 100]
            ),
            angularaxis=dict(
                showline=True,
                linecolor=font_color,
                tickfont=dict(color=font_color, size=12, family="Arial"),
                tickcolor=font_color
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=legend_font_color, family="Arial"),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=100, l=30, r=30, b=50),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font=dict(color=font_color, family="Arial"),
        template="none"
    )

    if titulo:
        fig.update_layout(title=dict(text= f"<b>{titulo}</b>", font=dict(size=18, color=font_color), x=0.5, xanchor='center'))
    if subtitulo:
        fig.add_annotation(
            text=f"<span style='font-size:12px; color:{annotation_color}'>{subtitulo}</span>",
            x=0.5, y=1.19, xref="paper", yref="paper",
            showarrow=False, xanchor="center"
        )

    return fig

# ===============================
# 🍕 PIZZA CHART (MPLSoccer)
# ===============================
# Función universal para las páginas 3, 4 y 5, unificando estilo, colores y lógica de percentiles.

@st.cache_resource
def crear_pizza_chart(jugador, df_contexto, metricas, considerar_dict, tipos_dict, metricas_invertir=None, titulo=None, subtitulo=None, modo_claro=None):
    """
    Crea un gráfico de pizza chart usando PyPizza con estilo unificado.
    - jugador: fila del jugador a graficar (Series)
    - df_contexto: base sobre la que se calculan los percentiles
    - metricas: lista de métricas a graficar
    - considerar_dict, tipos_dict: diccionarios auxiliares para color y filtrado
    - metricas_invertir: opcional, métricas que deben invertirse
    """

    percentiles = obtener_percentiles(jugador, df_contexto, metricas, metricas_invertir or [])
    colores = [
        "#6FBF73" if tipos_dict.get(m) == "Físicas"
        else "#2F80ED" if tipos_dict.get(m) == "Construcción"
        else "#EB5757" if tipos_dict.get(m) == "Ofensivas"
        else "#9B51E0"
        for m in metricas
    ]
    metricas_wrapped = [textwrap.fill(m, width=18, max_lines=2, placeholder="…") for m in metricas]

    background_color = "#FFFFFF" if modo_claro else "#0f1116"
    text_color = "#000000" if modo_claro else "#FFFFFF"
    value_color = "#FFFFFF"

    baker = PyPizza(
        params=metricas_wrapped,
        background_color=background_color,
        straight_line_color=text_color,
        straight_line_lw=1,
        last_circle_lw=1e-5,
        other_circle_lw=1e-5,
        inner_circle_size=20
    )

    fig, ax = baker.make_pizza(
        percentiles,
        figsize=(8, 8),
        color_blank_space="same",
        slice_colors=colores,
        value_bck_colors=colores,
        value_colors=[text_color] * len(percentiles),
        blank_alpha=0.2,
        param_location=110,
        kwargs_slices=dict(edgecolor=text_color, linewidth=1, zorder=2),
        kwargs_params=dict(color=text_color, fontsize=10, va="center"),
        kwargs_values=dict(color=value_color, fontsize=10, zorder=3,
                           bbox=dict(edgecolor=text_color, facecolor=background_color, boxstyle="round,pad=0.2", linewidth=1))
    )

    if titulo:
        fig.text(0.5, 1.00, titulo, size=16, ha="center", color=text_color, fontweight="bold")
    if subtitulo:
        fig.text(0.5, 0.97, subtitulo, size=13, ha="center", color=text_color)

    return fig

# ===============================
# ⚽ RADAR STATSBOMB (MPLSoccer)
# ===============================
# Similar al radar Percentil pero con estilo StatsBomb
# para uso en branding o publicaciones

@st.cache_resource
def crear_radar_statsbomb(jugador, df_contexto, seleccionadas, metricas_invertir, titulo, usar_percentiles=False, subtitulo=None, modo_claro=None):
    if usar_percentiles:
        percentiles = obtener_percentiles(jugador, df_contexto, seleccionadas, metricas_invertir)
        valores = [0 if pd.isnull(v) else v / 100 for v in percentiles]
        low = [0 for _ in seleccionadas]
        high = [1 for _ in seleccionadas]
    else:
        valores = jugador[seleccionadas].values.flatten().tolist()
        valores = [0 if pd.isnull(v) else v for v in valores]
        low = [df_contexto[m].replace(0, np.nan).min() for m in seleccionadas]
        high = [df_contexto[m].replace(0, np.nan).max() for m in seleccionadas]

    if modo_claro:
        facecolor_main = '#FFFFFF'
        facecolor_circles = '#DDDDDD'
        edgecolor_radar = '#666666'
        text_color = '#000000'
        facecolor_rings = '#CCCCCC'
    else:
        facecolor_main = '#0f1116'
        facecolor_circles = '#28252c'
        edgecolor_radar = '#5c5c5c'
        text_color = '#fcfcfc'
        facecolor_rings = '#0053a0'

    radar = Radar(params=seleccionadas, min_range=low, max_range=high, num_rings=3,
                  center_circle_radius=1)#, lower_is_better=[m for m in seleccionadas if m in metricas_invertir])

    fig, ax = grid(figheight=3.0, grid_height=0.9, title_height=0.08, endnote_height=0.0, grid_key='radar', axis=False)[0:2]
    radar.setup_axis(ax=ax['radar'], facecolor=facecolor_main)
    radar.draw_circles(ax=ax['radar'], facecolor=facecolor_circles, edgecolor='#39353f' if not modo_claro else '#999999')

    color_primario = jugador['Color primario'] if pd.notnull(jugador.get('Color primario')) else '#e63946'
    color_secundario = jugador['Color secundario'] if pd.notnull(jugador.get('Color secundario')) else '#0053a0'

    radar.draw_radar(valores, ax=ax['radar'],
                     kwargs_radar={'facecolor': color_primario, 'alpha': 1, 'edgecolor': edgecolor_radar, 'linewidth': 1},
                     kwargs_rings={'facecolor': color_secundario, 'alpha': 1})

    radar.draw_param_labels(ax=ax['radar'], fontsize=4.5, color=text_color)
    radar.draw_range_labels(ax=ax['radar'], fontsize=4, color=text_color)
    ax['title'].text(0.5, 0.7, titulo, fontsize=7, ha='center', color=text_color, fontweight='bold')
    fig.set_facecolor(facecolor_main)

    # Añadir subtítulo y pies de página
    if subtitulo:
        ax['title'].text(0.5, 0.32, subtitulo, fontsize=5, ha='center', color=text_color)

    #ax['title'].text(0.01, -11.7, "Minutos jugados / M90s:", fontsize=5, ha='left', color=text_color)
    ax['title'].text(0.99, -11.7, "Elaborado por: Jurgen Schmidt", fontsize=5, ha='right', color=text_color)

    return fig