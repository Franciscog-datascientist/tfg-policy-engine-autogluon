"""
ollama_reporter.py — Generador de informes con LLM local (Ollama)

Recibe los resultados del entrenamiento y las notas del policy engine
y genera un informe en lenguaje natural usando un modelo local vía Ollama.

El LLM no toma decisiones — solo redacta y explica lo que ya decidió
el policy engine y lo que obtuvieron los modelos entrenados.
"""

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"


def generate_report(
    dataset_meta: dict,
    target_meta: dict,
    user_goals: dict,
    config: dict,
    training_results: dict,
) -> str:
    """
    Genera un informe en lenguaje natural sobre el proceso de entrenamiento.

    Parámetros
    ----------
    dataset_meta      → características del dataset (n_rows, n_cols, etc.)
    target_meta       → características de la variable objetivo
    user_goals        → preferencias del usuario (priority, time_budget, etc.)
    config            → configuración generada por el policy engine
    training_results  → resultados del entrenamiento (score, best_model, etc.)

    Retorna
    -------
    str — informe en texto plano en español
    """

    # --- Extraer información relevante ---
    n_rows       = dataset_meta["n_rows"]
    n_cols       = dataset_meta["n_cols"]
    missing      = dataset_meta["missing_ratio"] * 100
    problem_type = config["predictor_init"]["problem_type"]
    eval_metric  = config["predictor_init"]["eval_metric"]
    preset       = config["fit_args"]["presets"][0]
    time_limit   = config["fit_args"]["time_limit"]
    notes        = config["notes"]
    best_model   = training_results["best_model"]
    score        = abs(training_results["score"])
    train_time   = training_results["training_time"]
    n_models     = len(training_results["leaderboard"])
    label        = user_goals["label"]
    priority     = user_goals["priority"]

    # Formatear score según métrica
    metric_labels = {
        "accuracy":                "exactitud global (accuracy)",
        "balanced_accuracy":       "exactitud equilibrada (balanced accuracy)",
        "f1":                      "F1",
        "f1_macro":                "F1 macro",
        "root_mean_squared_error": "error cuadrático medio (RMSE)",
    }
    problem_labels = {
        "binary":     "clasificación binaria",
        "multiclass": "clasificación multiclase",
        "regression": "regresión",
    }
    preset_labels = {
        "medium_quality": "calidad media (rápido)",
        "good_quality":   "buena calidad (equilibrado)",
        "high_quality":   "alta calidad (exhaustivo)",
        "best_quality":   "máxima calidad",
    }
    priority_labels = {
        "speed":       "velocidad",
        "balanced":    "equilibrio entre velocidad y rendimiento",
        "performance": "máximo rendimiento",
    }

    if eval_metric in ("accuracy", "balanced_accuracy", "f1", "f1_macro"):
        score_str = f"{score * 100:.2f}%"
    else:
        score_str = f"{score:.4f}"

    notes_text = "\n".join(f"- {n}" for n in notes)

    # --- Construir el prompt ---
    prompt = f"""Eres un asistente de ciencia de datos que genera informes claros y profesionales en español.
Escribe un informe conciso (máximo 250 palabras) explicando los resultados de un proceso de entrenamiento automático de modelos de machine learning.
Usa un lenguaje claro, sin tecnicismos innecesarios. No uses listas de puntos, escribe en prosa.
No inventes información — basa el informe únicamente en los datos que se te proporcionan.

DATOS DEL DATASET:
- Filas: {n_rows} | Columnas: {n_cols} | Valores nulos: {missing:.1f}%
- Variable objetivo: {label}

TIPO DE PROBLEMA: {problem_labels.get(problem_type, problem_type)}
PRIORIDAD DEL USUARIO: {priority_labels.get(priority, priority)}

DECISIONES DEL SISTEMA:
{notes_text}

CONFIGURACIÓN USADA:
- Estrategia: {preset_labels.get(preset, preset)}
- Tiempo límite: {time_limit // 60} minutos
- Métrica de evaluación: {metric_labels.get(eval_metric, eval_metric)}

RESULTADOS:
- Modelos entrenados: {n_models}
- Mejor modelo: {best_model}
- Puntuación final ({metric_labels.get(eval_metric, eval_metric)}): {score_str}
- Tiempo real de entrenamiento: {train_time}s

Escribe el informe ahora, en español, en prosa, sin títulos ni listas:"""

    # --- Llamar a Ollama ---
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "⚠️ No se pudo conectar con Ollama. "
            "Asegúrate de que Ollama está en ejecución (`ollama serve`) "
            "e inténtalo de nuevo."
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️ El modelo tardó demasiado en responder. "
            "Inténtalo de nuevo o usa un modelo más ligero."
        )
    except Exception as e:
        return f"⚠️ Error al generar el informe: {e}"