"""
policy_engine.py — Motor de decisión para AutoGluon (V1)

Orquesta el pipeline completo:
    1. Carga el dataset (una sola vez)
    2. Llama a analyze_dataset(df) → dataset_meta
    3. Llama a analyze_target(df, label) → target_meta
    4. Recibe user_goals (preferencias del usuario)
    5. Aplica las reglas de decisión
    6. Devuelve la configuración de AutoGluon + notes + run_id

Reglas de decisión implementadas (V1):
    - problem_type:  binary | multiclass | regression
    - eval_metric:   accuracy | balanced_accuracy | f1 | f1_macro | root_mean_squared_error
    - presets:       medium_quality | good_quality | high_quality | best_quality
                     + modificador optimize_for_deployment
    - time_limit:    300 | 1800 | 7200 segundos
    - calibrate_decision_threshold: True | False (solo binario)
    - notes:         lista de mensajes explicativos

Mecanismo de memoria (V1):
    - Cada ejecución se registra en execution_log.json con un run_id único.
    - Los resultados del entrenamiento se añaden al log tras entrenar,
      llamando a update_execution_log_with_results().
    - query_similar_runs() permite consultar ejecuciones pasadas similares
      para mostrar sugerencias al usuario antes de entrenar.

Referencias:
    - He & Garcia (2009) para umbrales de desbalanceo
    - Erickson et al. (2020) para heurística de problem_type
    - Hollmann et al. (2024) para límite de dataset pequeño
    - Documentación de AutoGluon (2024) para presets y calibrate_decision_threshold
"""

import json
import os
from datetime import datetime

import pandas as pd
from dataset_analyzer import analyze_dataset
from target_analyzer import analyze_target


## Excepción para caso ambiguo de problem_type

class AmbiguousProblemTypeError(Exception):
    """
    Se lanza cuando el tipo de problema no puede determinarse automáticamente.

    Ocurre cuando target_dtype es numérico y target_n_unique / n_rows <= 0.05,
    ya que en este rango la ambigüedad entre regresión y clasificación es real
    y una heurística rígida podría inducir errores.

    El usuario debe confirmar el tipo de problema pasando el parámetro
    confirmed_problem_type ('regression' o 'multiclass') a la función run().
    """
    pass


## Validación de entradas del usuario

_VALID_PRIORITIES = {"performance", "balanced", "speed"}
_VALID_TIME_BUDGETS = {"low", "medium", "high"}
_VALID_YES_NO = {"yes", "no"}
_VALID_PROBLEM_TYPES = {"binary", "multiclass", "regression"}


def _validate_user_goals(user_goals: dict) -> None:
    """
    Valida que las entradas del usuario tengan valores permitidos.

    Raises:
        ValueError — si algún campo tiene un valor no permitido.
    """
    required = ["label", "priority", "time_budget_level",
                 "focus_minority_class", "deployment_needed"]
    for field in required:
        if field not in user_goals:
            raise ValueError(f"Falta el campo obligatorio '{field}' en user_goals.")

    if user_goals["priority"] not in _VALID_PRIORITIES:
        raise ValueError(
            f"priority debe ser uno de {_VALID_PRIORITIES}, "
            f"se recibió '{user_goals['priority']}'."
        )

    if user_goals["time_budget_level"] not in _VALID_TIME_BUDGETS:
        raise ValueError(
            f"time_budget_level debe ser uno de {_VALID_TIME_BUDGETS}, "
            f"se recibió '{user_goals['time_budget_level']}'."
        )

    if user_goals["focus_minority_class"] not in _VALID_YES_NO:
        raise ValueError(
            f"focus_minority_class debe ser 'yes' o 'no', "
            f"se recibió '{user_goals['focus_minority_class']}'."
        )

    if user_goals["deployment_needed"] not in _VALID_YES_NO:
        raise ValueError(
            f"deployment_needed debe ser 'yes' o 'no', "
            f"se recibió '{user_goals['deployment_needed']}'."
        )



# Reglas para decidir problem_type

def _decide_problem_type(
    target_meta: dict,
    n_rows: int,
    notes: list,
    confirmed_problem_type: str | None = None,
) -> str:
    """
    Determina el tipo de problema según las reglas del documento (V1).

    Orden de prioridad:
        1. Si confirmed_problem_type está definido, se usa directamente.
        2. target_n_unique == 2 → binary
        3. target_dtype categórico/booleano y target_n_unique > 2 → multiclass
        4. target_dtype numérico y target_n_unique / n_rows > 0.05 → regression
        5. Caso ambiguo → AmbiguousProblemTypeError

    Referencia: Erickson et al. (2020) para la heurística del 0.05.
    """
    # Caso 0: el usuario ha confirmado manualmente
    if confirmed_problem_type is not None:
        if confirmed_problem_type not in _VALID_PROBLEM_TYPES:
            raise ValueError(
                f"confirmed_problem_type debe ser uno de {_VALID_PROBLEM_TYPES}, "
                f"se recibió '{confirmed_problem_type}'."
            )
        notes.append(
            f"Tipo de problema confirmado manualmente por el usuario: "
            f"{confirmed_problem_type}."
        )
        return confirmed_problem_type

    target_dtype = target_meta["target_dtype"]
    target_n_unique = target_meta["target_n_unique"]

    # Regla 1: exactamente 2 valores únicos → binario
    if target_n_unique == 2:
        notes.append("Se detecta clasificación binaria (target_n_unique = 2).")
        return "binary"

    # Regla 2: categórico/booleano con más de 2 clases → multiclase
    if target_dtype in ("categorical", "boolean") and target_n_unique > 2:
        notes.append(
            f"Se detecta clasificación multiclase "
            f"(target_dtype = {target_dtype}, target_n_unique = {target_n_unique})."
        )
        return "multiclass"

    # Regla 3: numérico con alta proporción de valores únicos → regresión
    if target_dtype == "numeric" and target_n_unique / n_rows > 0.05:
        notes.append(
            f"Se detecta regresión "
            f"(target_dtype = numeric, target_n_unique / n_rows = "
            f"{round(target_n_unique / n_rows, 4)} > 0.05). "
            f"Heurística inspirada en AutoGluon (Erickson et al., 2020)."
        )
        return "regression"

    # Caso ambiguo: numérico con baja proporción de únicos
    if target_dtype == "numeric" and target_n_unique / n_rows <= 0.05:
        ratio = round(target_n_unique / n_rows, 4)
        raise AmbiguousProblemTypeError(
            f"No se puede determinar automáticamente el tipo de problema. "
            f"La variable objetivo es numérica con {target_n_unique} valores "
            f"únicos sobre {n_rows} filas (ratio = {ratio} ≤ 0.05). "
            f"Esto podría ser regresión o clasificación multiclase. "
            f"Por favor, indique el tipo de problema pasando "
            f"confirmed_problem_type='regression' o "
            f"confirmed_problem_type='multiclass' a la función run()."
        )

    # Fallback (no debería llegar aquí con los dtypes actuales)
    raise AmbiguousProblemTypeError(
        f"Tipo de dato no reconocido: target_dtype = {target_dtype}, "
        f"target_n_unique = {target_n_unique}. "
        f"Confirme manualmente con confirmed_problem_type."
    )


# Reglas para decidir eval_metric

def _decide_eval_metric(
    problem_type: str,
    target_meta: dict,
    focus_minority_class: str,
    notes: list,
) -> str:
    """
    Selecciona la métrica de evaluación según las reglas del documento (V1).

    Binario:
        - focus_minority_class = yes → f1
        - focus_minority_class = no  y imbalance_ratio > 1.5 → balanced_accuracy
        - focus_minority_class = no  y imbalance_ratio ≤ 1.5 → accuracy

    Multiclase:
        - imbalance_ratio > 1.5 → f1_macro
        - imbalance_ratio ≤ 1.5 → accuracy

    Regresión:
        - root_mean_squared_error (siempre)
    """
    imbalance_ratio = target_meta.get("imbalance_ratio")

    # Binario 
    if problem_type == "binary":
        if focus_minority_class == "yes":
            notes.append(
                "Se selecciona f1 como métrica porque el usuario prioriza "
                "el rendimiento sobre la clase minoritaria."
            )
            return "f1"

        if imbalance_ratio is not None and imbalance_ratio > 1.5:
            notes.append(
                f"Se selecciona balanced_accuracy por desbalanceo de clases "
                f"(imbalance_ratio = {imbalance_ratio} > 1.5)."
            )
            return "balanced_accuracy"

        notes.append(
            f"Se selecciona accuracy (dataset equilibrado, "
            f"imbalance_ratio = {imbalance_ratio})."
        )
        return "accuracy"

    # Multiclase
    if problem_type == "multiclass":
        if imbalance_ratio is not None and imbalance_ratio > 1.5:
            notes.append(
                f"Se selecciona f1_macro por desbalanceo de clases "
                f"(imbalance_ratio = {imbalance_ratio} > 1.5)."
            )
            return "f1_macro"

        notes.append(
            f"Se selecciona accuracy (dataset equilibrado, "
            f"imbalance_ratio = {imbalance_ratio})."
        )
        return "accuracy"

    # Regresión
    if problem_type == "regression":
        notes.append(
            "Se selecciona root_mean_squared_error como métrica por defecto "
            "para regresión."
        )
        return "root_mean_squared_error"

    # Fallback (no debería llegar aquí)
    raise ValueError(f"problem_type no reconocido: {problem_type}")


# Reglas para decidir presets

def _decide_presets(
    priority: str,
    time_budget_level: str,
    deployment_needed: str,
    n_rows: int,
    notes: list,
) -> list:
    """
    Selecciona el preset de AutoGluon según las reglas del documento (V1).

    Regla prioritaria:
        - n_rows < 1000 → medium_quality (máximo), con nota explicativa.

    Tabla de decisión (priority × time_budget_level):
        - speed  + low/medium  → medium_quality
        - speed  + high        → good_quality
        - balanced (cualquier) → good_quality
        - performance + low    → high_quality (con advertencia)
        - performance + medium → high_quality
        - performance + high   → best_quality

    Modificador:
        - deployment_needed = yes → se añade optimize_for_deployment
    """
    small_dataset = n_rows < 1000

    # Determinar preset base según priority × time_budget_level 
    if priority == "speed":
        if time_budget_level in ("low", "medium"):
            preset = "medium_quality"
            notes.append(
                "Se selecciona medium_quality (prioridad = velocidad, "
                f"presupuesto temporal = {time_budget_level})."
            )
        else:  # high
            preset = "good_quality"
            notes.append(
                "Se eleva el preset a good_quality porque el presupuesto "
                "temporal alto lo permite, a pesar de priorizar velocidad."
            )

    elif priority == "balanced":
        preset = "good_quality"
        notes.append(
            "Se selecciona good_quality por equilibrio entre "
            "rendimiento y coste."
        )

    elif priority == "performance":
        if time_budget_level == "low":
            preset = "high_quality"
            notes.append(
                "Se selecciona high_quality (prioridad = rendimiento). "
                "Advertencia: el presupuesto temporal es bajo (300s), "
                "lo que puede limitar la exploración de modelos."
            )
        elif time_budget_level == "medium":
            preset = "high_quality"
            notes.append(
                "Se selecciona high_quality (prioridad = rendimiento, "
                "presupuesto temporal = medio)."
            )
        else:  # high
            preset = "best_quality"
            notes.append(
                "Se selecciona best_quality (prioridad = rendimiento, "
                "presupuesto temporal = alto)."
            )

    # Regla prioritaria: dataset pequeño
    if small_dataset:
        preset = "medium_quality"
        notes.append(
            "Dataset pequeño detectado (n_rows < 1000). Se limita el "
            "preset a medium_quality para reducir el riesgo de "
            "sobreajuste, siguiendo las recomendaciones de "
            "Hollmann et al. (2024)."
        )

    # Construir lista de presets 
    presets = [preset]

    if deployment_needed == "yes":
        presets.append("optimize_for_deployment")
        notes.append(
            "Se añade optimize_for_deployment por necesidad de despliegue."
        )

    return presets


# Reglas para decidir time_limit

def _decide_time_limit(time_budget_level: str, notes: list) -> int:
    """
    Asigna el tiempo máximo de entrenamiento según las reglas del documento (V1).

    Mapeo:
        - low    → 300 segundos  (5 minutos)
        - medium → 1800 segundos (30 minutos)
        - high   → 7200 segundos (2 horas)
    """
    mapping = {
        "low": 300,
        "medium": 1800,
        "high": 7200,
    }

    time_limit = mapping[time_budget_level]
    notes.append(
        f"Se fija time_limit = {time_limit}s por selección de "
        f"presupuesto temporal {time_budget_level}."
    )
    return time_limit


# Reglas para calibrate_decision_threshold

def _decide_calibrate_threshold(
    problem_type: str,
    eval_metric: str,
    notes: list,
) -> bool:
    """
    Decide si calibrar el umbral de decisión según las reglas del documento (V1).

    Solo aplica a problemas binarios con f1 o balanced_accuracy,
    donde la documentación de AutoGluon indica que puede mejorar
    significativamente el rendimiento de estas métricas.
    """
    if problem_type == "binary" and eval_metric in ("f1", "balanced_accuracy"):
        notes.append(
            f"Se activa calibrate_decision_threshold (problem_type = binary, "
            f"eval_metric = {eval_metric})."
        )
        return True

    return False


# Log de ejecuciones (memoria)

_LOG_FILE = "execution_log.json"


def _load_log() -> list:
    """Carga el log de ejecuciones desde disco. Devuelve lista vacía si no existe."""
    if os.path.exists(_LOG_FILE):
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_log(log: list) -> None:
    """Escribe el log completo en disco."""
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def _save_execution_log(
    dataset_meta: dict,
    target_meta: dict,
    user_goals: dict,
    config: dict,
    run_id: str,
) -> None:
    """
    Guarda una entrada nueva en el log acumulativo de ejecuciones.

    Cada entrada registra:
        - run_id:        identificador único de la ejecución (timestamp ISO)
        - dataset_meta:  características del dataset
        - target_meta:   características de la variable objetivo
        - user_goals:    preferencias del usuario
        - config:        configuración generada por el policy engine
        - training_results: None hasta que se llame a update_execution_log_with_results()
    """
    entry = {
        "run_id": run_id,
        "timestamp": run_id,           # alias legible
        "dataset_meta": dataset_meta,
        "target_meta": target_meta,
        "user_goals": user_goals,
        "config": config,
        "training_results": None,      # se rellenará tras entrenar
    }

    log = _load_log()
    log.append(entry)
    _save_log(log)


def update_execution_log_with_results(run_id: str, training_results: dict) -> bool:
    """
    Actualiza la entrada del log identificada por run_id con los resultados
    del entrenamiento.

    Parámetros
    ----------
    run_id : str
        Identificador de la ejecución devuelto por run() dentro de config["run_id"].
    training_results : dict
        Diccionario serializable con los resultados del entrenamiento.
        Estructura esperada (preparada en app.py antes de llamar a esta función):
        {
            "best_model":       str,   nombre técnico del mejor modelo
            "score":            float, puntuación en test externo
            "eval_metric":      str,   métrica utilizada
            "training_time":    float, segundos reales de entrenamiento
            "leaderboard_top5": [      top 5 modelos con sus scores
                {"model": str, "score_val": float | None, "score_test": float | None},
                ...
            ]
        }

    Retorna
    -------
    bool
        True si se encontró y actualizó la entrada, False si no se encontró.
    """
    log = _load_log()

    for entry in log:
        if entry.get("run_id") == run_id:
            entry["training_results"] = training_results
            _save_log(log)
            return True

    return False  # run_id no encontrado en el log


def _size_bucket(n_rows: int) -> str:
    """
    Clasifica el número de filas en un bucket de tamaño para comparación
    de similaridad entre datasets.

    Buckets:
        tiny   → < 1 000
        small  → 1 000 – 9 999
        medium → 10 000 – 99 999
        large  → ≥ 100 000
    """
    if n_rows < 1_000:
        return "tiny"
    if n_rows < 10_000:
        return "small"
    if n_rows < 100_000:
        return "medium"
    return "large"


def query_similar_runs(
    problem_type: str,
    n_rows: int,
    top_k: int = 3,
) -> list[dict]:
    """
    Busca en el log las ejecuciones más relevantes completadas (con resultados
    de entrenamiento guardados) que sean similares al problema actual.

    Criterios de similitud:
        1. Mismo problem_type  (obligatorio)
        2. Mismo bucket de tamaño de dataset  (obligatorio)

    Dentro de los candidatos, devuelve los top_k ordenados por mejor score:
        - Clasificación: mayor score primero (accuracy/f1 más alto)
        - Regresión:     menor error absoluto primero (RMSE más bajo)

    Parámetros
    ----------
    problem_type : str
        Tipo de problema del run actual ('binary', 'multiclass', 'regression').
    n_rows : int
        Número de filas del dataset actual.
    top_k : int
        Número máximo de resultados a devolver.

    Retorna
    -------
    list[dict]
        Lista de hasta top_k entradas con los campos relevantes para mostrar
        al usuario. Cada elemento contiene:
        {
            "run_id":        str,
            "timestamp":     str,
            "n_rows":        int,
            "eval_metric":   str,
            "preset":        str,
            "best_model":    str,
            "score":         float,
            "training_time": float,
        }
        Lista vacía si no hay runs similares completados.
    """
    log = _load_log()
    bucket = _size_bucket(n_rows)
    is_regression = (problem_type == "regression")

    candidates = []
    for entry in log:
        # Debe tener resultados de entrenamiento guardados
        if entry.get("training_results") is None:
            continue

        # Mismo tipo de problema
        entry_problem_type = entry.get("config", {}).get(
            "predictor_init", {}
        ).get("problem_type")
        if entry_problem_type != problem_type:
            continue

        # Mismo bucket de tamaño
        entry_n_rows = entry.get("dataset_meta", {}).get("n_rows", 0)
        if _size_bucket(entry_n_rows) != bucket:
            continue

        tr = entry["training_results"]
        score = tr.get("score")
        if score is None:
            continue

        preset_list = entry.get("config", {}).get("fit_args", {}).get("presets", [])
        preset = preset_list[0] if preset_list else "—"

        candidates.append({
            "run_id":        entry["run_id"],
            "timestamp":     entry.get("timestamp", entry["run_id"]),
            "n_rows":        entry_n_rows,
            "eval_metric":   tr.get("eval_metric", "—"),
            "preset":        preset,
            "best_model":    tr.get("best_model", "—"),
            "score":         score,
            "training_time": tr.get("training_time", 0),
        })

    if not candidates:
        return []

    # Ordenar: regresión → menor |score| primero; clasificación → mayor score primero
    candidates.sort(
        key=lambda x: abs(x["score"]) if is_regression else -x["score"]
    )

    return candidates[:top_k]


# Función principal: run()

def run(
    source: pd.DataFrame | str,
    label: str,
    priority: str,
    time_budget_level: str,
    focus_minority_class: str,
    deployment_needed: str,
    confirmed_problem_type: str | None = None,
) -> dict:
    """
    Ejecuta el pipeline completo del policy engine (V1).

    Parámetros
    ----------
    source : pd.DataFrame | str
        DataFrame ya cargado o ruta a un archivo CSV.
    label : str
        Nombre de la columna objetivo.
    priority : str
        Prioridad del usuario: 'performance', 'balanced' o 'speed'.
    time_budget_level : str
        Presupuesto temporal: 'low', 'medium' o 'high'.
    focus_minority_class : str
        Atención a la clase minoritaria: 'yes' o 'no'.
        Solo relevante en clasificación; se ignora en regresión.
    deployment_needed : str
        Necesidad de despliegue: 'yes' o 'no'.
    confirmed_problem_type : str | None
        Si el tipo de problema es ambiguo, el usuario puede confirmarlo
        pasando 'regression' o 'multiclass'. Por defecto None.

    Retorna
    -------
    dict
        Configuración de AutoGluon con la estructura:
        {
            "run_id": str,              ← identificador único de esta ejecución
            "predictor_init": {
                "label": str,
                "problem_type": str,
                "eval_metric": str
            },
            "fit_args": {
                "presets": list[str],
                "time_limit": int
            },
            "post_fit": {
                "calibrate_decision_threshold": bool
            },
            "notes": list[str]
        }

    Raises
    ------
    AmbiguousProblemTypeError
        Si el tipo de problema no se puede determinar automáticamente
        y no se ha proporcionado confirmed_problem_type.
    ValueError
        Si algún parámetro tiene un valor no permitido.
    """

    # Validar entradas del usuario 
    user_goals = {
        "label": label,
        "priority": priority,
        "time_budget_level": time_budget_level,
        "focus_minority_class": focus_minority_class,
        "deployment_needed": deployment_needed,
    }
    _validate_user_goals(user_goals)

    # Cargar dataset (una sola vez)
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, str):
        df = pd.read_csv(source)
    else:
        raise TypeError(
            f"source debe ser un DataFrame o una ruta (str), "
            f"se recibió {type(source).__name__}."
        )

    # Análisis del dataset (Bloque 1 de entradas)
    dataset_meta = analyze_dataset(df)

    # Análisis del target (Bloque 3 de entradas)
    target_meta = analyze_target(df, label)

    # Aplicar reglas de decisión
    notes = []

    # 1. problem_type
    problem_type = _decide_problem_type(
        target_meta=target_meta,
        n_rows=dataset_meta["n_rows"],
        notes=notes,
        confirmed_problem_type=confirmed_problem_type,
    )

    # 2. eval_metric
    eval_metric = _decide_eval_metric(
        problem_type=problem_type,
        target_meta=target_meta,
        focus_minority_class=focus_minority_class,
        notes=notes,
    )

    # 3. presets
    presets = _decide_presets(
        priority=priority,
        time_budget_level=time_budget_level,
        deployment_needed=deployment_needed,
        n_rows=dataset_meta["n_rows"],
        notes=notes,
    )

    # 4. time_limit
    time_limit = _decide_time_limit(
        time_budget_level=time_budget_level,
        notes=notes,
    )

    # 5. calibrate_decision_threshold
    calibrate = _decide_calibrate_threshold(
        problem_type=problem_type,
        eval_metric=eval_metric,
        notes=notes,
    )

    # Generar run_id único para esta ejecución
    run_id = datetime.now().isoformat()

    # Construir salida
    config = {
        "run_id": run_id,
        "predictor_init": {
            "label": label,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
        },
        "fit_args": {
            "presets": presets,
            "time_limit": time_limit,
        },
        "post_fit": {
            "calibrate_decision_threshold": calibrate,
        },
        "notes": notes,
    }

    # Guardar ejecución en el log de memoria
    _save_execution_log(
        dataset_meta=dataset_meta,
        target_meta=target_meta,
        user_goals=user_goals,
        config=config,
        run_id=run_id,
    )

    return config



# Ejecución directa para pruebas rápidas

if __name__ == "__main__":
    import json
    import numpy as np

    # Dataset de ejemplo
    df_example = pd.DataFrame({
        "edad": [25, 30, None, 45, 22],
        "salario": [30000.0, 45000.0, 50000.0, None, 28000.0],
        "ciudad": ["Madrid", "Valencia", "Madrid", None, "Sevilla"],
        "activo": [True, False, True, True, False],
        "target": [1, 0, 1, 0, 1],
    })

    print("=" * 60)
    print("POLICY ENGINE — Prueba con dataset de ejemplo")
    print("=" * 60)

    config = run(
        source=df_example,
        label="target",
        priority="balanced",
        time_budget_level="medium",
        focus_minority_class="no",
        deployment_needed="no",
    )

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\nrun_id: {config['run_id']}")

    # Simular que el entrenamiento ha terminado
    fake_results = {
        "best_model": "WeightedEnsemble_L2",
        "score": 0.82,
        "eval_metric": "accuracy",
        "training_time": 45.3,
        "leaderboard_top5": [
            {"model": "WeightedEnsemble_L2", "score_val": 0.81, "score_test": 0.82},
            {"model": "LightGBM_BAG_L1",     "score_val": 0.79, "score_test": None},
        ],
    }

    updated = update_execution_log_with_results(config["run_id"], fake_results)
    print(f"\nResultados guardados en el log: {updated}")

    # --- Consultar runs similares ---
    similar = query_similar_runs(
        problem_type=config["predictor_init"]["problem_type"],
        n_rows=5,
    )
    print(f"\nRuns similares encontrados: {len(similar)}")
    for r in similar:
        print(f"  {r['timestamp'][:19]} | {r['best_model']} | score={r['score']}")