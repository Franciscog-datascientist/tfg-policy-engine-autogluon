"""
trainer.py — Módulo de entrenamiento con AutoGluon

Recibe la configuración generada por el policy engine y ejecuta
AutoGluon TabularPredictor con los parámetros decididos.

Flujo:
    1. Recibe el DataFrame y la config del policy engine
    2. Divide en train/test
    3. Crea TabularPredictor con predictor_init
    4. Ejecuta fit() con fit_args
    5. Aplica calibrate_decision_threshold si corresponde
    6. Evalúa en test y devuelve resultados + métricas adicionales

Retorna un diccionario con:
    - leaderboard: ranking de modelos entrenados
    - best_model: nombre del mejor modelo
    - score: puntuación en test con la métrica seleccionada
    - eval_metric: métrica utilizada
    - training_time: tiempo real de entrenamiento (segundos)
    - extra_metrics: métricas adicionales (accuracy, precision, recall, f1)
                     solo para problemas de clasificación
    - model_path: ruta donde AutoGluon guardó los modelos
"""

import time
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def train(
    df: pd.DataFrame,
    config: dict,
    test_size: float = 0.2,
    random_seed: int = 67,
    save_path: str = "AutogluonModels",
) -> dict:
    """
    Entrena un modelo con AutoGluon usando la configuración del policy engine.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo (se dividirá internamente en train/test).
    config : dict
        Configuración generada por policy_engine.run().
    test_size : float
        Proporción del dataset reservada para test (por defecto 0.2).
    random_seed : int
        Semilla para reproducibilidad en la división train/test.
    save_path : str
        Directorio donde AutoGluon guardará los modelos entrenados.

    Retorna
    -------
    dict con leaderboard, best_model, score, eval_metric, training_time,
         extra_metrics, model_path y predictor.
    """

    # --- Extraer configuración ---
    label        = config["predictor_init"]["label"]
    problem_type = config["predictor_init"]["problem_type"]
    eval_metric  = config["predictor_init"]["eval_metric"]
    presets      = config["fit_args"]["presets"]
    time_limit   = config["fit_args"]["time_limit"]
    calibrate    = config["post_fit"]["calibrate_decision_threshold"]

    # --- Dividir en train/test ---
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx   = int(len(df_shuffled) * (1 - test_size))
    train_data  = df_shuffled.iloc[:split_idx]
    test_data   = df_shuffled.iloc[split_idx:]

    print(f"Train: {len(train_data)} filas | Test: {len(test_data)} filas")
    print(f"Label: {label} | Tipo: {problem_type} | Métrica: {eval_metric}")
    print(f"Presets: {presets} | Time limit: {time_limit}s")
    print("=" * 60)

    # --- Crear predictor ---
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=save_path,
    )

    # --- Entrenar ---
    start_time = time.time()
    predictor.fit(
        train_data=train_data,
        presets=presets,
        time_limit=time_limit,
    )
    training_time = round(time.time() - start_time, 2)

    # --- Calibrar umbral si corresponde ---
    if calibrate:
        print("\nCalibrando umbral de decisión...")
        predictor.calibrate_decision_threshold()

    # --- Obtener predicciones sobre test ---
    y_true = test_data[label]
    y_pred = predictor.predict(test_data)

    # --- Calcular score principal ---
    try:
        score_result = predictor.evaluate(test_data)
        if isinstance(score_result, dict):
            score = score_result.get(eval_metric, list(score_result.values())[0])
        else:
            score = score_result
    except Exception as e:
        print(f"\nAviso: evaluate() falló ({e}). Evaluando manualmente...")
        metric_functions = {
            "accuracy":                lambda yt, yp: accuracy_score(yt, yp),
            "balanced_accuracy":       lambda yt, yp: balanced_accuracy_score(yt, yp),
            "f1":                      lambda yt, yp: f1_score(yt, yp, average="binary"),
            "f1_macro":                lambda yt, yp: f1_score(yt, yp, average="macro"),
            "root_mean_squared_error": lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)),
        }
        if eval_metric in metric_functions:
            score = round(metric_functions[eval_metric](y_true, y_pred), 4)
        else:
            score = round(accuracy_score(y_true, y_pred), 4)

    # --- Calcular métricas adicionales ---
    # Para clasificación: accuracy, precision, recall, f1
    # Para regresión: MAE, R²
    extra_metrics = {}

    if problem_type in ("binary", "multiclass"):
        avg = "binary" if problem_type == "binary" else "macro"
        try:
            kwargs = {"average": avg, "zero_division": 0}
            if problem_type == "binary":
                # Detectar la clase positiva desde los datos reales
                # (puede ser 0/1 o strings como 'Yes'/'No')
                unique_labels = sorted(y_true.unique())
                kwargs["pos_label"] = unique_labels[-1]
            extra_metrics = {
                "accuracy":  round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, **kwargs), 4),
                "recall":    round(recall_score(y_true, y_pred, **kwargs), 4),
                "f1":        round(f1_score(y_true, y_pred, **kwargs), 4),
            }
        except Exception as e:
            print(f"Aviso: no se pudieron calcular métricas adicionales: {e}")

    elif problem_type == "regression":
        try:
            extra_metrics = {
                "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
                "mae":  round(mean_absolute_error(y_true, y_pred), 4),
                "r2":   round(r2_score(y_true, y_pred), 4),
            }
        except Exception as e:
            print(f"Aviso: no se pudieron calcular métricas adicionales: {e}")

    leaderboard = predictor.leaderboard(test_data, silent=True, extra_info=True)
    best_model  = predictor.model_best

    # --- Mostrar resultados ---
    print("\n" + "=" * 60)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"Mejor modelo: {best_model}")
    print(f"Score ({eval_metric}): {score}")
    print(f"Tiempo de entrenamiento: {training_time}s")
    if extra_metrics:
        print(f"Métricas adicionales: {extra_metrics}")
    print(f"\nLeaderboard (top 5):")
    print(leaderboard.head().to_string(index=False))

    return {
        "leaderboard":    leaderboard,
        "best_model":     best_model,
        "score":          score,
        "eval_metric":    eval_metric,
        "training_time":  training_time,
        "extra_metrics":  extra_metrics,
        "predictor":      predictor,
    }


# ============================================================================
# Ejecución directa para prueba rápida
# ============================================================================
if __name__ == "__main__":
    from policy_engine import run

    np.random.seed(42)

    n = 200
    df_example = pd.DataFrame({
        "edad":       np.random.randint(18, 65, n),
        "salario":    np.random.randint(20000, 80000, n),
        "experiencia":np.random.randint(0, 30, n),
        "ciudad":     np.random.choice(["Madrid", "Valencia", "Barcelona"], n),
        "target":     np.random.choice([0, 1], n),
    })

    config = run(
        source=df_example,
        label="target",
        priority="speed",
        time_budget_level="low",
        focus_minority_class="no",
        deployment_needed="no",
    )

    results = train(df_example, config)
    print("\nMétricas extra:", results["extra_metrics"])