"""
app.py — Interfaz Streamlit para el Policy Engine de AutoGluon (V1)

Flujo de pasos:
    0 → El usuario sube el CSV
    1 → Análisis automático del dataset (dataset_meta)
    2 → El usuario elige la columna objetivo (label)
    3 → Análisis de la variable objetivo (target_meta) + inferencia de problem_type
        → Si es ambiguo, se pregunta al usuario (paso 3b)
    4 → Preguntas guiadas al usuario (priority, time_budget_level, deployment_needed,
        focus_minority_class — esta última solo si NO es regresión)
    5 → El policy engine genera el JSON de configuración y se muestra el resultado

Cómo ejecutar:
    streamlit run app.py

Requisitos:
    pip install streamlit pandas
    (policy_engine.py, dataset_analyzer.py y target_analyzer.py deben
    estar en el mismo directorio que este archivo)
"""

import io as _io
import json
import os
import re
import zipfile

import pandas as pd
import streamlit as st

from policy_engine import (
    run,
    AmbiguousProblemTypeError,
    update_execution_log_with_results,  # ← NUEVO: guardar resultados en el log
    query_similar_runs,                 # ← NUEVO: consultar ejecuciones similares
)
from dataset_analyzer import analyze_dataset
from target_analyzer import analyze_target
from trainer import train
from ollama_reporter import generate_report


# =============================================================================
# Configuración general de la página
# =============================================================================

st.set_page_config(
    page_title="Policy Engine — AutoGluon",
    page_icon=None,
    layout="centered",
)

st.markdown(
    """
    <style>
        /* Fondo blanco general */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        section.main,
        .block-container {
            background-color: #ffffff !important;
        }

        /* Todo el texto negro */
        html, body, p, li,
        h1, h2, h3, h4, h5, h6,
        label, .stMarkdown, .stText {
            color: #000000 !important;
        }

        /* Spans y divs: solo color, no tocar backgrounds de componentes */
        .stMarkdown span,
        .stMarkdown div,
        .stMarkdown p {
            color: #000000 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploadDropzone"] * {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stFileUploadDropzone"] {
            border: 1.5px solid #aaaaaa !important;
            border-radius: 4px !important;
        }
        [data-testid="stFileUploadDropzone"] svg,
        [data-testid="stFileUploadDropzone"] svg * {
            fill: #000000 !important;
            stroke: none !important;
        }
        [data-testid="stFileUploadDropzone"] button {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
            border: 1px solid #aaaaaa !important;
        }

        /* Botones generales: fondo blanco, borde negro */
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1.5px solid #000000 !important;
            border-radius: 4px !important;
        }
        .stButton > button:hover {
            background-color: #f0f0f0 !important;
        }
        /* Boton primario: negro con texto blanco
           Streamlit añade data-testid="baseButton-primary" al boton primario */
        [data-testid="baseButton-primary"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1.5px solid #000000 !important;
        }
        [data-testid="baseButton-primary"]:hover {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
        [data-testid="baseButton-primary"] p,
        [data-testid="baseButton-primary"] span {
            color: #ffffff !important;
        }
        /* Boton secundario */
        [data-testid="baseButton-secondary"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1.5px solid #000000 !important;
        }
        [data-testid="baseButton-secondary"] p,
        [data-testid="baseButton-secondary"] span {
            color: #000000 !important;
        }

        /* Alertas / info / success / warning */
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div {
            background-color: #f9f9f9 !important;
            border: 1px solid #aaaaaa !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] div {
            color: #000000 !important;
        }

        /* Expanders */
        [data-testid="stExpander"],
        details, summary {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stExpander"] * {
            color: #000000 !important;
        }

        /* Metricas */
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"] {
            color: #000000 !important;
        }

        /* Selectbox — control visible */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-color: #aaaaaa !important;
        }
        /* Selectbox dropdown popover */
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        [data-baseweb="menu"],
        [data-baseweb="menu"] *,
        [role="listbox"],
        [role="listbox"] *,
        [role="option"],
        [role="option"] * {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #000000 !important;
        }
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background-color: #f0f0f0 !important;
        }

        /* Radio buttons: solo texto, no el indicador grafico */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p {
            color: #000000 !important;
        }

        /* Inputs de texto */
        .stTextInput input,
        .stNumberInput input {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-color: #aaaaaa !important;
        }

        /* Code blocks */
        code, pre, .stCode,
        [data-testid="stCode"] {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
        }

        /* JSON viewer */
        [data-testid="stJson"],
        [data-testid="stJson"] * {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Dataframe / tabla — fondo blanco y texto negro */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        [data-testid="stDataFrameResizable"],
        [data-testid="stDataFrameResizable"] *,
        .dvn-scroller,
        .dvn-scroller * {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        /* Cabecera del dataframe */
        [role="columnheader"],
        [role="columnheader"] * {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
        }
        /* Celdas */
        [role="gridcell"],
        [role="gridcell"] * {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [role="row"]:hover [role="gridcell"] {
            background-color: #f5f5f5 !important;
        }

        /* Captions */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: #555555 !important;
        }

        /* Divisores */
        hr {
            border-color: #cccccc !important;
        }

        /* JSON viewer */
        [data-testid="stJson"],
        [data-testid="stJson"] *,
        .stJson,
        .stJson * {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #000000 !important;
        }

        /* Spinner */
        [data-testid="stSpinner"] * {
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Inicialización de session_state
# =============================================================================

defaults = {
    "step": 0,
    "df": None,
    "dataset_meta": None,
    "label": None,
    "target_meta": None,
    "problem_type_auto": None,
    "confirmed_problem_type": None,
    "priority": None,
    "time_budget_level": None,
    "deployment_needed": None,
    "focus_minority_class": None,
    "config": None,
    "run_id": None,                 # ← NUEVO: identificador único de la ejecución
    "training_results": None,
    "results_saved_to_log": False,  # ← NUEVO: flag para no guardar dos veces
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# Helpers
# =============================================================================

def reset():
    """Reinicia toda la sesión para volver al paso 0."""
    for key in defaults:
        st.session_state[key] = defaults[key]
    st.rerun()


def _imbalance_label(ratio: float | None) -> str:
    if ratio is None:
        return "No aplica (regresión)"
    if ratio <= 1.5:
        return f"{ratio} — Dataset equilibrado"
    if ratio < 10:
        return f"{ratio} — Desbalanceo ligero"
    return f"{ratio} — Desbalanceo moderado/severo"


def _infer_problem_type(target_meta: dict, n_rows: int) -> str | None:
    dtype = target_meta["target_dtype"]
    n_unique = target_meta["target_n_unique"]
    if n_unique == 2:
        return "binary"
    if dtype in ("categorical", "boolean") and n_unique > 2:
        return "multiclass"
    if dtype == "numeric" and (n_unique / n_rows) > 0.05:
        return "regression"
    return None


_MODEL_BASE_MAP = {
    "weightedensemble": ("", "Ensemble ponderado"),
    "lightgbmlarge":    ("", "LightGBM Grande"),
    "lightgbmxt":       ("", "LightGBM Extra Trees"),
    "lightgbm":         ("", "LightGBM"),
    "catboost":         ("", "CatBoost"),
    "xgboost":          ("", "XGBoost"),
    "randomforest":     ("", "Random Forest"),
    "extratrees":       ("", "Extra Trees"),
    "neuralnettorch":   ("", "Red neuronal (PyTorch)"),
    "neuralnetfastai":  ("", "Red neuronal (FastAI)"),
    "kneighbors":       ("", "K vecinos más cercanos"),
    "linearmodel":      ("", "Modelo lineal"),
}


def _friendly_model_name(raw: str) -> str:
    """Convierte el nombre técnico de AutoGluon en un nombre legible."""
    lower = raw.lower()
    emoji, label = "", raw
    for key, (e, l) in _MODEL_BASE_MAP.items():
        if key in lower:
            emoji, label = e, l
            break
    level_match = re.search(r"_l(\d+)", lower)
    level_str = f" · Nivel {level_match.group(1)}" if level_match else ""
    bag_str = " (bagging)" if "_bag_" in lower else ""
    return f"{emoji}{label}{bag_str}{level_str}".lstrip()


def _serialize_leaderboard(leaderboard: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """
    Convierte el leaderboard de AutoGluon en una lista de dicts serializable a JSON.
    Solo guarda los top_k modelos con sus scores numéricos.
    """
    records = []
    for _, row in leaderboard.head(top_k).iterrows():
        entry = {"model": str(row["model"])}
        if "score_val" in row.index:
            sv = row["score_val"]
            entry["score_val"] = None if pd.isna(sv) else float(sv)
        if "score_test" in row.index:
            st_val = row["score_test"]
            entry["score_test"] = None if pd.isna(st_val) else float(st_val)
        records.append(entry)
    return records


# =============================================================================
# Cabecera
# =============================================================================

st.title("Policy Engine para AutoGluon")
st.markdown(
    "Este asistente analiza tu dataset y genera automáticamente la "
    "configuración óptima de AutoGluon según tus objetivos."
)
st.divider()


# =============================================================================
# PASO 0 — Subir el CSV
# =============================================================================

if st.session_state.step == 0:

    st.subheader("Paso 1 de 5 — Sube tu dataset")
    st.markdown("Selecciona un archivo CSV. El sistema lo analizará automáticamente.")

    uploaded = st.file_uploader(
        label="Archivo CSV",
        type=["csv"],
        help="El archivo debe tener cabecera con nombres de columna.",
    )

    if uploaded is not None:
        with st.spinner("Cargando y analizando el dataset…"):
            try:
                df = pd.read_csv(uploaded)
                dataset_meta = analyze_dataset(df)

                st.session_state.df = df
                st.session_state.dataset_meta = dataset_meta
                st.session_state.step = 1

                st.rerun()

            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")


# =============================================================================
# PASO 1 — Mostrar resumen del dataset y elegir la columna objetivo
# =============================================================================

elif st.session_state.step == 1:

    df = st.session_state.df
    meta = st.session_state.dataset_meta

    st.subheader("Paso 2 de 5 — Elige la columna objetivo")

    st.markdown("**Resumen del dataset detectado:**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", meta["n_rows"])
    col2.metric("Columnas", meta["n_cols"])
    col3.metric("% valores nulos", f"{meta['missing_ratio']*100:.1f}%")

    col4, col5 = st.columns(2)
    col4.metric("Variables numéricas", meta["num_numeric_features"])
    col5.metric("Variables categóricas", meta["num_categorical_features"])

    st.divider()

    with st.expander("Ver primeras filas del dataset"):
        # Preview del dataset como HTML
        preview_df = df.head(10)
        header_html = "".join(f'<th style="padding:6px 10px;text-align:left;color:#000000;background:#f0f0f0;border-bottom:2px solid #cccccc;">{c}</th>' for c in preview_df.columns)
        rows_preview = ""
        for _, row in preview_df.iterrows():
            cells = "".join(f'<td style="padding:6px 10px;color:#000000;border-bottom:1px solid #eeeeee;">{v}</td>' for v in row)
            rows_preview += f"<tr>{cells}</tr>"
        st.markdown(f"""
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;background:#ffffff;font-size:13px;">
          <thead><tr>{header_html}</tr></thead>
          <tbody>{rows_preview}</tbody>
        </table></div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**¿Qué columna quieres predecir?**")
    st.caption(
        "Selecciona la variable objetivo (target). "
        "El resto de columnas se usarán como predictores."
    )

    label = st.selectbox(
        label="Columna objetivo (label)",
        options=df.columns.tolist(),
        index=None,
        placeholder="Selecciona una columna…",
    )

    col_btn, col_reset = st.columns([3, 1])

    with col_btn:
        if st.button("Continuar", disabled=(label is None), use_container_width=True):
            with st.spinner("Analizando la variable objetivo…"):
                target_meta = analyze_target(df, label)
                problem_type_auto = _infer_problem_type(target_meta, meta["n_rows"])
                st.session_state.label = label
                st.session_state.target_meta = target_meta
                st.session_state.problem_type_auto = problem_type_auto
                st.session_state.step = 2
                st.rerun()

    with col_reset:
        if st.button("Reiniciar", use_container_width=True):
            reset()


# =============================================================================
# PASO 2 — Mostrar análisis de la variable objetivo
# =============================================================================

elif st.session_state.step == 2:

    target_meta = st.session_state.target_meta
    problem_type_auto = st.session_state.problem_type_auto
    label = st.session_state.label

    st.subheader("Paso 3 de 5 — Análisis de la variable objetivo")
    st.markdown(f"**Columna seleccionada:** `{label}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tipo de dato", target_meta["target_dtype"])
    col2.metric("Valores únicos", target_meta["target_n_unique"])
    col3.metric("Columnas predictoras", target_meta["n_predictive_cols"])

    st.markdown(
        f"**Desbalanceo de clases:** {_imbalance_label(target_meta['imbalance_ratio'])}"
    )

    st.divider()

    if problem_type_auto is not None:
        type_labels = {
            "binary":     "Clasificacion binaria",
            "multiclass": "Clasificacion multiclase",
            "regression": "Regresion",
        }
        st.success(
            f"Tipo de problema detectado automáticamente: "
            f"**{type_labels[problem_type_auto]}**"
        )

        col_btn, col_reset = st.columns([3, 1])

        with col_btn:
            if st.button("Continuar", use_container_width=True):
                st.session_state.confirmed_problem_type = None
                st.session_state.step = 3
                st.rerun()

        with col_reset:
            if st.button("Reiniciar", use_container_width=True):
                reset()

    else:
        st.warning(
            f"No se puede determinar el tipo de problema automaticamente. "
            f"La columna `{label}` es numérica con pocos valores únicos relativos, "
            f"lo que puede ser tanto **regresión** como **clasificación multiclase**. "
            f"Por favor, indícalo manualmente."
        )
        st.markdown("**¿Cuál es el tipo de problema correcto?**")

        confirmed = st.radio(
            label="Tipo de problema",
            options=["multiclass", "regression"],
            format_func=lambda x: (
                "Clasificacion multiclase — la columna representa categorias numericas"
                if x == "multiclass"
                else "Regresion — la columna es un valor numerico continuo"
            ),
            index=None,
        )

        col_btn, col_reset = st.columns([3, 1])

        with col_btn:
            if st.button(
                "Confirmar y continuar",
                disabled=(confirmed is None),
                use_container_width=True,
            ):
                st.session_state.confirmed_problem_type = confirmed
                st.session_state.problem_type_auto = confirmed
                st.session_state.step = 3
                st.rerun()

        with col_reset:
            if st.button("Reiniciar", use_container_width=True):
                reset()


# =============================================================================
# PASO 3 — Preguntas guiadas al usuario
# =============================================================================

elif st.session_state.step == 3:

    problem_type = (
        st.session_state.confirmed_problem_type
        or st.session_state.problem_type_auto
    )
    is_regression   = (problem_type == "regression")
    target_meta     = st.session_state.target_meta
    imbalance_ratio = target_meta.get("imbalance_ratio")

    st.subheader("Paso 4 de 5 — Configura tus preferencias")
    st.markdown(
        "Responde las siguientes preguntas para que el sistema "
        "pueda decidir la mejor configuración de AutoGluon."
    )
    st.divider()

    # -------------------------------------------------------------------------
    # Pregunta 1: priority
    # -------------------------------------------------------------------------
    st.markdown("**¿Qué priorizas en el entrenamiento?**")
    priority = st.radio(
        label="Prioridad",
        options=["speed", "balanced", "performance"],
        format_func=lambda x: {
            "speed":       "Velocidad — Quiero resultados rapidos aunque sean menos precisos",
            "balanced":    "Equilibrio — Un buen balance entre tiempo y precision",
            "performance": "Rendimiento — Quiero el mejor modelo posible aunque tarde mas",
        }[x],
        index=None,
        label_visibility="collapsed",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Pregunta 2: time_budget_level (dependiente de priority)
    # -------------------------------------------------------------------------
    time_budget = None

    if priority is not None:
        time_options_map = {
            "speed":       ["low", "medium"],
            "balanced":    ["low", "medium", "high"],
            "performance": ["medium", "high"],
        }
        time_labels = {
            "low":    "Poco — Maximo 5 minutos (300s)",
            "medium": "Medio — Hasta 30 minutos (1800s)",
            "high":   "Mucho — Hasta 2 horas (7200s)",
        }
        available_times = time_options_map[priority]
        context_msg = {
            "speed":       "Como priorizas velocidad, el sistema limita el tiempo "
                           "máximo a 30 minutos para mantener la coherencia.",
            "balanced":    "Con prioridad equilibrada tienes disponibles todos "
                           "los rangos de tiempo.",
            "performance": "Como priorizas rendimiento, el tiempo mínimo disponible "
                           "es 30 minutos para que AutoGluon pueda explorar bien el espacio de modelos.",
        }

        st.markdown("**¿Cuánto tiempo estás dispuesto a esperar para entrenar?**")
        st.caption(context_msg[priority])

        time_budget = st.radio(
            label="Presupuesto temporal",
            options=available_times,
            format_func=lambda x: time_labels[x],
            index=None,
            label_visibility="collapsed",
        )

        st.divider()

    # -------------------------------------------------------------------------
    # Pregunta 3: deployment_needed
    # -------------------------------------------------------------------------
    st.markdown("**¿Vas a desplegar el modelo en producción?**")
    st.caption("Si es así, se favorecerán modelos más ligeros y rápidos en inferencia.")
    deployment = st.radio(
        label="Despliegue",
        options=["no", "yes"],
        format_func=lambda x: {
            "no":  "No — Solo quiero analizar resultados",
            "yes": "Si — Necesito un modelo listo para produccion",
        }[x],
        index=None,
        label_visibility="collapsed",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Pregunta 4: focus_minority_class (solo si clasificación con desbalanceo)
    # -------------------------------------------------------------------------
    focus_minority = "no"

    if not is_regression:
        if imbalance_ratio is not None and imbalance_ratio > 1.5:
            if imbalance_ratio >= 10:
                imbalance_desc = f"Desbalanceo severo (ratio = {imbalance_ratio})"
                imbalance_hint = (
                    "La clase más frecuente aparece más de 10 veces más que la "
                    "menos frecuente. Prestar atención a la clase minoritaria "
                    "puede ser clave para obtener un modelo útil."
                )
            else:
                imbalance_desc = f"Desbalanceo ligero (ratio = {imbalance_ratio})"
                imbalance_hint = (
                    f"La clase más frecuente aparece "
                    f"{imbalance_ratio:.1f} veces más que la menos frecuente."
                )

            st.markdown("**¿Quieres prestar atención especial a la clase minoritaria?**")
            st.info(
                f"**Desbalanceo detectado en tu dataset:** {imbalance_desc}\n\n"
                f"{imbalance_hint}"
            )

            focus_minority = st.radio(
                label="Clase minoritaria",
                options=["no", "yes"],
                format_func=lambda x: {
                    "no":  "No — Me interesa el rendimiento global del modelo",
                    "yes": "Sí — Quiero que el modelo rinda bien también en la clase minoritaria",
                }[x],
                index=None,
                label_visibility="collapsed",
            )
            st.divider()

        else:
            st.info(
                f"Tu dataset esta bien equilibrado "
                f"(imbalance_ratio = {imbalance_ratio if imbalance_ratio else 'N/A'}). "
                f"No es necesario ajustar el enfoque hacia la clase minoritaria."
            )

    # -------------------------------------------------------------------------
    # Validación y botón continuar
    # -------------------------------------------------------------------------
    all_answered = all([
        priority is not None,
        time_budget is not None,
        deployment is not None,
        focus_minority is not None,
    ])

    col_btn, col_reset = st.columns([3, 1])

    with col_btn:
        if st.button(
            "Generar configuracion",
            disabled=not all_answered,
            use_container_width=True,
        ):
            st.session_state.priority = priority
            st.session_state.time_budget_level = time_budget
            st.session_state.deployment_needed = deployment
            st.session_state.focus_minority_class = focus_minority
            st.session_state.step = 4
            st.rerun()

    with col_reset:
        if st.button("Reiniciar", use_container_width=True):
            reset()


# =============================================================================
# PASO 4 — Ejecutar el policy engine y mostrar el resultado
# =============================================================================

elif st.session_state.step == 4:

    # --- Calcular config solo la primera vez ---
    if st.session_state.config is None:
        with st.spinner("El policy engine está generando la configuración…"):
            try:
                config = run(
                    source=st.session_state.df,
                    label=st.session_state.label,
                    priority=st.session_state.priority,
                    time_budget_level=st.session_state.time_budget_level,
                    focus_minority_class=st.session_state.focus_minority_class,
                    deployment_needed=st.session_state.deployment_needed,
                    confirmed_problem_type=st.session_state.confirmed_problem_type,
                )
                st.session_state.config = config
                # ── NUEVO: guardar run_id en session_state ──────────────────
                st.session_state.run_id = config.get("run_id")

            except AmbiguousProblemTypeError as e:
                st.error(f"Error de tipo de problema ambiguo: {e}")
                st.stop()

            except Exception as e:
                st.error(f"Error inesperado en el policy engine: {e}")
                st.stop()

    config = st.session_state.config

    st.subheader("Paso 4 de 5 — Configuracion generada")
    st.success(
        "El sistema ha analizado tu dataset y ha decidido la configuración "
        "óptima de AutoGluon. Revísala y pulsa **Entrenar modelo** cuando estés listo."
    )
    st.divider()

    problem_type = config["predictor_init"]["problem_type"]
    eval_metric  = config["predictor_init"]["eval_metric"]
    presets_list = config["fit_args"]["presets"]
    time_limit   = config["fit_args"]["time_limit"]
    calibrate    = config["post_fit"]["calibrate_decision_threshold"]
    main_preset  = presets_list[0]
    has_deployment = "optimize_for_deployment" in presets_list

    problem_labels = {
        "binary":     "Clasificacion binaria",
        "multiclass": "Clasificacion multiclase",
        "regression": "Regresion",
    }
    preset_labels = {
        "medium_quality": "Rápido — calidad media",
        "good_quality":   "Equilibrado — buena calidad",
        "high_quality":   "Exhaustivo — alta calidad",
        "best_quality":   "Máximo — mejor calidad posible",
    }
    metric_labels = {
        "accuracy":                "Exactitud global (accuracy)",
        "balanced_accuracy":       "Exactitud equilibrada (balanced accuracy)",
        "f1":                      "F1 — equilibrio entre precisión y recall",
        "f1_macro":                "F1 macro — rendimiento medio entre clases",
        "root_mean_squared_error": "Error cuadrático medio (RMSE)",
        "mean_absolute_error":     "Error absoluto medio (MAE)",
    }

    st.markdown("**Resumen de lo que va a hacer el sistema:**")
    st.markdown(
        f"- **Tipo de problema:** {problem_labels.get(problem_type, problem_type)}\n"
        f"- **Metrica de evaluacion:** {metric_labels.get(eval_metric, eval_metric)}\n"
        f"- **Estrategia de entrenamiento:** {preset_labels.get(main_preset, main_preset)}\n"
        f"- **Tiempo maximo:** {time_limit // 60} minutos ({time_limit}s)\n"
        + ("- **Optimizado para despliegue en produccion**\n" if has_deployment else "")
        + ("- **Se ajustara el umbral de decision automaticamente**\n" if calibrate else "")
    )

    st.divider()

    with st.expander("¿Por qué esta configuración? — Ver razonamiento del sistema"):
        for i, note in enumerate(config["notes"], start=1):
            st.markdown(f"{i}. {note}")

    with st.expander("Ver configuración técnica completa (JSON)"):
        st.json(config)

    st.divider()

    # ── NUEVO: Ejecuciones similares del historial ────────────────────────────
    # Solo se muestra si hay al menos una ejecución previa completada similar.
    # Se consulta por problem_type y tamaño del dataset para no mezclar
    # contextos incomparables (ej. regresión con clasificación).
    n_rows = st.session_state.dataset_meta["n_rows"]
    similar_runs = query_similar_runs(
        problem_type=problem_type,
        n_rows=n_rows,
        top_k=3,
    )

    if similar_runs:
        with st.expander(
            f"Historial — {len(similar_runs)} ejecucion(es) similar(es) encontrada(s)",
            expanded=False,
        ):
            st.caption(
                "Ejecuciones anteriores con el mismo tipo de problema y un dataset "
                "de tamaño comparable. Pueden darte una referencia del rendimiento esperable."
            )

            is_regression_view = (problem_type == "regression")

            for run_entry in similar_runs:
                fecha = run_entry["timestamp"][:19].replace("T", " ")
                model_friendly = _friendly_model_name(run_entry["best_model"])
                score_val = run_entry["score"]

                if is_regression_view:
                    score_str = f"Error: {abs(score_val):.4f}"
                    score_icon = ""
                else:
                    score_str = f"Score: {abs(score_val) * 100:.2f}%"
                    score_icon = ""

                st.markdown(
                    f"**{fecha}** · {score_str} · "
                    f"Mejor modelo: {model_friendly} · "
                    f"Preset: `{run_entry['preset']}` · "
                    f"Métrica: `{run_entry['eval_metric']}` · "
                    f"Tiempo: {run_entry['training_time']:.1f}s"
                )

    # ─────────────────────────────────────────────────────────────────────────

    col_train, col_reset = st.columns([3, 1])

    with col_train:
        if st.button(
            "Entrenar modelo",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.step = 5
            st.rerun()

    with col_reset:
        if st.button("Reiniciar", use_container_width=True):
            reset()


# =============================================================================
# PASO 5 — Entrenar con AutoGluon y mostrar resultados
# =============================================================================

elif st.session_state.step == 5:

    config = st.session_state.config

    # --- Entrenar solo si no tenemos resultados ya calculados ---
    if st.session_state.training_results is None:

        time_limit = config["fit_args"]["time_limit"]
        minutes    = time_limit // 60

        with st.spinner(
            f"Entrenando modelos... esto puede tardar hasta {minutes} minutos. "
            f"Puedes dejar la pantalla abierta y volver cuando termine."
        ):
            try:
                results = train(
                    df=st.session_state.df,
                    config=config,
                )
                st.session_state.training_results = {
                    "best_model":     results["best_model"],
                    "score":          results["score"],
                    "eval_metric":    results["eval_metric"],
                    "training_time":  results["training_time"],
                    "leaderboard":    results["leaderboard"],
                    "model_path":     results["predictor"].path,
                    "extra_metrics":  results.get("extra_metrics", {}),
                }

                # ── NUEVO: guardar resultados en execution_log.json ─────────
                # Construimos una versión serializable (sin DataFrame ni predictor)
                # y la vinculamos a la ejecución actual usando el run_id.
                if st.session_state.run_id is not None:
                    leaderboard_top5 = _serialize_leaderboard(results["leaderboard"])
                    serializable_results = {
                        "best_model":       results["best_model"],
                        "score":            float(results["score"]),
                        "eval_metric":      results["eval_metric"],
                        "training_time":    float(results["training_time"]),
                        "leaderboard_top5": leaderboard_top5,
                    }
                    saved = update_execution_log_with_results(
                        run_id=st.session_state.run_id,
                        training_results=serializable_results,
                    )
                    st.session_state.results_saved_to_log = saved
                # ─────────────────────────────────────────────────────────────

            except Exception as e:
                st.error(
                    f"Error durante el entrenamiento: {e}\n\n"
                    "Comprueba que AutoGluon está instalado: "
                    "`pip install autogluon`"
                )
                if st.button("Volver a la configuracion"):
                    st.session_state.step = 4
                    st.rerun()
                st.stop()

    results      = st.session_state.training_results
    problem_type = config["predictor_init"]["problem_type"]
    eval_metric  = results["eval_metric"]

    metric_labels = {
        "accuracy":                "Exactitud global (accuracy)",
        "balanced_accuracy":       "Exactitud equilibrada (balanced accuracy)",
        "f1":                      "F1 — equilibrio entre precisión y recall",
        "f1_macro":                "F1 macro — rendimiento medio entre clases",
        "root_mean_squared_error": "Error cuadrático medio (RMSE)",
        "mean_absolute_error":     "Error absoluto medio (MAE)",
    }

    # ── Pre-calcular el mejor modelo por media val+test ──────────────────────
    _lb_pre       = results["leaderboard"].copy()
    _has_test_pre = "score_test" in _lb_pre.columns

    if _has_test_pre:
        _lb_test = _lb_pre[_lb_pre["score_test"].notna()]
        if not _lb_test.empty:
            _best_pre_score = _lb_test["score_test"].abs().max()
        else:
            _best_pre_score = abs(results["score"])
    else:
        _best_pre_score = abs(results["score"])

    score         = _best_pre_score
    display_score = abs(score)

    # -------------------------------------------------------------------------
    # Cabecera de resultados
    # -------------------------------------------------------------------------
    st.subheader("Paso 5 de 5 — Resultados del entrenamiento")
    st.success("El entrenamiento ha finalizado correctamente.")

    if st.session_state.results_saved_to_log:
        st.caption("Resultados guardados en el historial de ejecuciones.")

    st.divider()

    # -------------------------------------------------------------------------
    # Métrica principal
    # -------------------------------------------------------------------------
    st.markdown("### ¿Cómo de bueno es el modelo?")

    if eval_metric in ("accuracy", "balanced_accuracy", "f1", "f1_macro"):
        pct = display_score * 100
        if pct >= 90:
            quality = "Excelente"
            quality_desc = "El modelo acierta en la gran mayoría de los casos."
        elif pct >= 75:
            quality = "Bueno"
            quality_desc = "El modelo funciona bien, con margen de mejora."
        elif pct >= 60:
            quality = "Aceptable"
            quality_desc = "El modelo aprende del dataset pero comete bastantes errores."
        else:
            quality = "Mejorable"
            quality_desc = (
                "El modelo tiene dificultades. Considera revisar el dataset "
                "o aumentar el tiempo de entrenamiento."
            )

        col1, col2 = st.columns([1, 2])
        col1.metric(label=f"Puntuación ({eval_metric})", value=f"{pct:.1f}%")
        col2.markdown(f"**{quality}**")
        col2.markdown(quality_desc)

        extra = results.get("extra_metrics", {})
        if extra:
            st.markdown("**Métricas detalladas del mejor modelo:**")
            report_df = pd.DataFrame([{
                "Métrica":     "Exactitud (Accuracy)",
                "Valor":       f"{extra.get('accuracy', 0) * 100:.1f}%",
                "Descripción": "Porcentaje de predicciones correctas sobre el total",
            }, {
                "Métrica":     "Precisión (Precision)",
                "Valor":       f"{extra.get('precision', 0) * 100:.1f}%",
                "Descripción": "De los positivos predichos, ¿cuántos eran realmente positivos?",
            }, {
                "Métrica":     "Sensibilidad (Recall)",
                "Valor":       f"{extra.get('recall', 0) * 100:.1f}%",
                "Descripción": "De todos los positivos reales, ¿cuántos detectó el modelo?",
            }, {
                "Métrica":     "F1 Score",
                "Valor":       f"{extra.get('f1', 0) * 100:.1f}%",
                "Descripción": "Equilibrio entre precisión y sensibilidad",
            }])
            # HTML table para garantizar visibilidad
            metrics_rows = ""
            for _, row in report_df.iterrows():
                metrics_rows += (
                    f'<tr style="border-bottom:1px solid #dddddd;">'
                    f'<td style="padding:8px 12px;font-weight:bold;color:#000000;">{row["Métrica"]}</td>'
                    f'<td style="padding:8px 12px;color:#000000;">{row["Valor"]}</td>'
                    f'<td style="padding:8px 12px;color:#555555;">{row["Descripción"]}</td>'
                    f'</tr>'
                )
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;background:#ffffff;font-size:14px;">
              <thead>
                <tr style="background:#f0f0f0;border-bottom:2px solid #cccccc;">
                  <th style="padding:8px 12px;text-align:left;color:#000000;">Métrica</th>
                  <th style="padding:8px 12px;text-align:left;color:#000000;">Valor</th>
                  <th style="padding:8px 12px;text-align:left;color:#000000;">Descripción</th>
                </tr>
              </thead>
              <tbody>{metrics_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            acc = extra.get("accuracy", 0)
            f1  = extra.get("f1", 0)
            if acc > 0.95 and f1 < 0.90 and eval_metric != "accuracy":
                st.warning(
                    "**Atencion: la exactitud (accuracy) puede ser enganosa en este caso.** "
                    "Cuando el dataset está muy desbalanceado, un modelo que prediga "
                    "siempre la clase mayoritaria obtiene una accuracy muy alta sin "
                    "detectar realmente los casos minoritarios. "
                    f"Por eso el sistema eligió **{metric_labels.get(eval_metric, eval_metric)}** "
                    "como métrica principal, que refleja mejor el rendimiento real."
                )

            st.caption(
                f"El modelo fue optimizado usando "
                f"**{metric_labels.get(eval_metric, eval_metric)}** "
                f"como criterio principal."
            )

    else:
        # RMSE u otras métricas de regresión
        col1, col2 = st.columns([1, 2])
        col1.metric(
            label=f"Error medio ({metric_labels.get(eval_metric, eval_metric)})",
            value=f"{display_score:.4f}",
            help="En regresión, cuanto más bajo, mejor.",
        )
        col2.markdown("**Regresión — error en la escala original de la variable objetivo.**")
        col2.markdown(
            "Un error bajo indica que las predicciones se acercan mucho al valor real."
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Información del entrenamiento
    # -------------------------------------------------------------------------
    st.markdown("### Detalles del entrenamiento")

    col1, col2 = st.columns(2)
    col1.metric("Tiempo de entrenamiento", f"{results['training_time']}s")
    col2.metric("Modelos probados", len(results["leaderboard"]))

    st.divider()

    # -------------------------------------------------------------------------
    # Leaderboard
    # -------------------------------------------------------------------------
    st.markdown("### Modelos probados (ranking)")

    leaderboard = results["leaderboard"].copy()

    is_classification = eval_metric in ("accuracy", "balanced_accuracy", "f1", "f1_macro")

    def _fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{abs(v) * 100:.2f}%" if is_classification else f"{abs(v):.4f}"

    # ── Mejor modelo = mayor score_test (métrica honesta sobre datos nuevos) ─
    has_test = "score_test" in leaderboard.columns
    has_val  = "score_val"  in leaderboard.columns

    if has_test:
        # Filtrar modelos que tienen score_test válido
        lb_with_test = leaderboard[leaderboard["score_test"].notna()]
        if not lb_with_test.empty:
            best_model_name = lb_with_test.loc[lb_with_test["score_test"].abs().idxmax(), "model"]
        else:
            best_model_name = results["best_model"]
    else:
        best_model_name = results["best_model"]

    best_row_data       = leaderboard[leaderboard["model"] == best_model_name].iloc[0]
    best_model_friendly = _friendly_model_name(best_model_name)
    best_test = abs(best_row_data["score_test"]) if has_test and not pd.isna(best_row_data.get("score_test", float("nan"))) else None
    best_val  = abs(best_row_data["score_val"])  if has_val  and not pd.isna(best_row_data.get("score_val",  float("nan"))) else None

    partes = []
    if best_test is not None: partes.append(f"**{_fmt(best_test)} en test externo** (datos no vistos durante el entrenamiento)")
    if best_val  is not None: partes.append(f"{_fmt(best_val)} en validación interna")
    st.info(f"**El mejor modelo es {best_model_friendly}** — " + " · ".join(partes) + ".")

    caption_base = (
        "Val (interno): score de validación cruzada durante el entrenamiento. "
        "Test (externo): score sobre el 20% de datos reservados antes de entrenar. "
        "El mejor modelo se elige por mayor score en test externo."
    )
    caption_full_note = (
        " Los modelos que muestran — en val son versiones _FULL: AutoGluon los reentrenó "
        "sobre el 100% de los datos para maximizar el rendimiento final, por lo que no tienen "
        "score de validación interna. Su score en test externo sigue siendo completamente válido."
    )

    # ── Construir tabla ──────────────────────────────────────────────────────
    leaderboard["Modelo"] = leaderboard["model"].apply(_friendly_model_name)

    def _build_score_str(row):
        val_str  = _fmt(row.get("score_val",  float("nan"))) if has_val  else "—"
        test_str = _fmt(row.get("score_test", float("nan"))) if has_test else "—"
        return f"{val_str} (val)  ·  {test_str} (test)"

    leaderboard["Val / Test"] = leaderboard.apply(_build_score_str, axis=1)
    leaderboard["Mejor"]      = leaderboard["model"].apply(
        lambda x: "Mejor" if x == best_model_name else ""
    )

    # ── Top 5 garantizando que el mejor aparece ──────────────────────────────
    # Ordenar por score_test desc, luego score_val desc
    sort_cols = []
    if has_test: sort_cols.append("score_test")
    if has_val:  sort_cols.append("score_val")
    if sort_cols:
        leaderboard = leaderboard.sort_values(sort_cols, ascending=False, na_position="last")

    if best_model_name in leaderboard.head(5)["model"].values:
        top_n = leaderboard.head(5)
    else:
        best_row  = leaderboard[leaderboard["model"] == best_model_name]
        other_top = leaderboard[leaderboard["model"] != best_model_name].head(4)
        top_n     = pd.concat([best_row, other_top])

    # Renderizar como HTML para garantizar fondo blanco y texto negro
    def _render_table(df):
        td_s = "padding:8px 12px;"
        td_b = "padding:8px 12px;font-weight:bold;"
        rows_html = ""
        for _, row in df.iterrows():
            modelo = row["Modelo"]
            vt = row["Val / Test"]
            mejor = row["Mejor"]
            mejor_td = (
                "<td style='" + td_b + "'>" + mejor + "</td>"
                if mejor else
                "<td style='" + td_s + "'></td>"
            )
            rows_html += (
                "<tr style='border-bottom:1px solid #dddddd;'>"
                "<td style='" + td_s + "'>" + modelo + "</td>"
                "<td style='" + td_s + "'>" + vt + "</td>"
                + mejor_td +
                "</tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;background:#ffffff;"
            "color:#000000;font-size:14px;'>"
            "<thead><tr style='background:#f0f0f0;border-bottom:2px solid #cccccc;'>"
            "<th style='padding:8px 12px;text-align:left;'>Modelo</th>"
            "<th style='padding:8px 12px;text-align:left;'>Val / Test</th>"
            "<th style='padding:8px 12px;text-align:left;'>Mejor</th>"
            "</tr></thead>"
            "<tbody>" + rows_html + "</tbody></table>"
        )
    some_full = has_val and top_n["score_val"].isna().any()
    st.caption(caption_base + (caption_full_note if some_full else ""))
    st.markdown(_render_table(top_n), unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------------------------------
    # Mensaje final orientativo
    # -------------------------------------------------------------------------
    st.markdown("### ¿Qué puedo hacer ahora con este modelo?")
    st.info(
        "El modelo entrenado ha sido guardado automáticamente en la carpeta "
        "`AutogluonModels/` dentro del directorio donde ejecutaste la aplicación. "
        "Puedes cargarlo en cualquier momento con AutoGluon para hacer predicciones "
        "sobre nuevos datos sin necesidad de volver a entrenar."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Descarga del modelo como .zip
    # -------------------------------------------------------------------------
    st.markdown("### Descargar modelo entrenado")
    st.markdown(
        "Descarga el modelo entrenado para usarlo más adelante sin necesidad "
        "de volver a entrenar. Una vez descargado puedes cargarlo con:"
    )
    st.code(
        "from autogluon.tabular import TabularPredictor\n"
        'predictor = TabularPredictor.load("AutogluonModels/")\n'
        "predictor.predict(nuevos_datos)",
        language="python",
    )

    model_path = results.get("model_path", "AutogluonModels")

    if os.path.exists(model_path):
        zip_buffer = _io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname   = os.path.relpath(file_path, start=os.path.dirname(model_path))
                    zf.write(file_path, arcname)
        zip_buffer.seek(0)

        st.download_button(
            label="Descargar modelo (.zip)",
            data=zip_buffer,
            file_name="modelo_autogluon.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
    else:
        st.warning(
            f"No se encontró la carpeta del modelo en `{model_path}`. "
            "Comprueba que el entrenamiento finalizó correctamente."
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Informe Ollama
    # -------------------------------------------------------------------------
    st.markdown("### Informe de resultados")
    with st.spinner("Generando informe de resultados…"):
        report = generate_report(
            dataset_meta=st.session_state.dataset_meta,
            target_meta=st.session_state.target_meta,
            user_goals={
                "label":                st.session_state.label,
                "priority":             st.session_state.priority,
                "time_budget_level":    st.session_state.time_budget_level,
                "focus_minority_class": st.session_state.focus_minority_class,
                "deployment_needed":    st.session_state.deployment_needed,
            },
            config=st.session_state.config,
            training_results=results,
        )

    st.markdown(report)

    st.divider()
    col_new, col_reset = st.columns([3, 1])

    with col_new:
        if st.button("Entrenar con otro dataset", use_container_width=True):
            reset()

    with col_reset:
        if st.button("Volver a la configuracion", use_container_width=True):
            st.session_state.step = 4
            st.session_state.training_results = None
            st.rerun()