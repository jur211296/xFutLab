import streamlit as st

# Helper seguro para detectar el tema aunque la versión de Streamlit no sea 1.46+
def get_theme_type():
    # Intento 1: API nueva (>=1.46)
    ctx = getattr(st, "context", None)
    if ctx is not None and hasattr(ctx, "theme"):
        theme = getattr(ctx, "theme", None)
        if theme is not None:
            t = getattr(theme, "type", None)
            if t in ("light", "dark"):
                return t
    # Intento 2: opción de configuración (themes en config.toml)
    base = st.get_option("theme.base")
    if base in ("light", "dark"):
        return base
    # Fallback
    return "light"

# ---- Página "Inicio" vacía (lienzo en blanco por ahora) ----
def pagina_inicio():
    # Detecta tema actual ('light' o 'dark') para uso futuro
    theme_type = get_theme_type()
    # No mostramos nada aún en el lienzo central, lo dejamos vacío.
    # Puedes usar 'theme_type' más adelante para condicionar estilos o gráficos.
    # st.write(f"(debug) Tema actual: {theme_type}")
    st.title("🏠 Bienvenido a xFutLab by Jürgen Schmidt")
    st.write("Usa el menú superior para navegar entre módulos.")

# ---- Declaración de páginas para la navegación superior ----
pages = {
    "Inicio": [
        st.Page(pagina_inicio, title="Inicio", icon="🏠", default=True),
    ],
    "Estadísticas": [
        st.Page("pages/1_Comparativa_general.py", title="📊 Comparativa general"),
        st.Page("pages/1_Comparativa_con_logos.py", title="🛡️ Comparativa con logos"),
        st.Page("pages/1_Comparativa_personalizada.py", title="📈 Comparativa personalizada"),
        st.Page("pages/1_Comparativa_multiple_radar.py", title="⚔️ Comparativa múltiple"),
    ],
    "1v1": [
        st.Page("pages/2_Comparativa_radar.py", title="⚔️ 1v1 - Radar"),
        st.Page("pages/2_Comparativa_statsbomb.py", title="🧨 1v1 - Statsbomb"),
        st.Page("pages/2_Comparativa_pizza.py", title="🍕 1v1 - Pizza"),
        st.Page("pages/2_Comparativa_scatter.py", title="📉 1v1 - Muestras"),
    ],
    "Análisis individual": [
        st.Page("pages/3_Perfil_individual_statsbomb.py", title="🧨 Perfil - Statsbomb"),
        st.Page("pages/3_Perfil_individual_pizza.py", title="🍕 Perfil - Pizza"),
    ],
    #"Scouting": [
    #    st.Page("pages/4_🔎_Búsqueda_avanzada.py", title="Búsqueda avanzada"),
    #],
}
# ---- Top Navigation (Streamlit 1.46+) ----
pg = st.navigation(pages, position="top")
pg.run()

        
        
        