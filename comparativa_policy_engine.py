"""
comparativa_policy_engine.py
============================
Compara dos configuraciones de AutoGluon frente al policy engine real:

    BASELINE_A — AutoGluon mínimo (medium_quality, 300s, accuracy/rmse)
                 Simula un usuario sin experiencia que lanza AutoGluon
                 con los parámetros por defecto.

    BASELINE_B — AutoGluon mínimo con el MISMO tiempo que el policy engine
                 (medium_quality, mismo time_limit, accuracy/rmse)
                 Aísla el efecto de la configuración inteligente del motor
                 eliminando la ventaja de tiempo.

    POLICY ENGINE — Configuración decidida automáticamente por el motor
                    a partir de las características del dataset y las
                    respuestas predefinidas del usuario.

Ambos se evalúan con las DOS métricas para comparativa completa:
    - La métrica del baseline (accuracy / root_mean_squared_error)
    - La métrica que decide el policy engine

Uso:
    py -3.11 comparativa_policy_engine.py

El script guarda:
    - comparativa_resultados.csv   → tabla completa de resultados
    - comparativa_resumen.txt      → resumen legible por consola

Coloca este script en la misma carpeta que policy_engine.py,
dataset_analyzer.py, target_analyzer.py y los CSV.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor
from policy_engine import AmbiguousProblemTypeError
from policy_engine import run as policy_engine_run

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN DE DATASETS
# =============================================================================

DATASETS = [
    {
        "name": "Titanic",
        "csv_path": "titanic.csv",
        "label": "Survived",
        "description": "Clasificación binaria equilibrada (supervivencia en el Titanic)",
        "baseline_metric": "accuracy",
        "confirmed_problem_type": None,
        "user_answers": {
            "priority": "performance",
            "time_budget_level": "medium",
            "focus_minority_class": "no",
            "deployment_needed": "no",
        },
    },
    {
        "name": "IBM HR Attrition",
        "csv_path": "ibm_attrition.csv",
        "label": "Attrition",
        "description": "Clasificación binaria con desbalanceo moderado (IR ≈ 5.2)",
        "baseline_metric": "accuracy",
        "confirmed_problem_type": None,
        "user_answers": {
            "priority": "performance",
            "time_budget_level": "medium",
            "focus_minority_class": "yes",
            "deployment_needed": "no",
        },
    },
    {
        "name": "Credit Card Fraud",
        "csv_path": "creditcard.csv",
        "label": "Class",
        "description": "Clasificación binaria con desbalanceo severo (IR ≈ 577)",
        "baseline_metric": "accuracy",
        "confirmed_problem_type": None,
        "user_answers": {
            "priority": "performance",
            "time_budget_level": "medium",
            "focus_minority_class": "yes",
            "deployment_needed": "no",
        },
    },
    {
        "name": "Loan Approval",
        "csv_path": "loan_approval.csv",
        "label": "approved",
        "description": "Clasificación binaria con desbalanceo ligero (IR ≈ 1.6)",
        "baseline_metric": "accuracy",
        "confirmed_problem_type": None,
        "user_answers": {
            "priority": "balanced",
            "time_budget_level": "medium",
            "focus_minority_class": "yes",
            "deployment_needed": "no",
        },
    },
    {
        "name": "Video Game Sales",
        "csv_path": "Video_Games_Sales_Cleaned.csv",
        "label": "total_sales",
        "description": "Regresión sobre ventas totales de videojuegos",
        "baseline_metric": "root_mean_squared_error",
        "confirmed_problem_type": "regression",
        "user_answers": {
            "priority": "balanced",
            "time_budget_level": "medium",
            "focus_minority_class": "no",
            "deployment_needed": "no",
        },
    },
]

# Configuración del baseline A (usuario sin experiencia)
BASELINE_PRESET    = "medium_quality"
BASELINE_TIME_A    = 300   # 5 minutos — usuario sin experiencia


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def evaluate_manual(y_true, y_pred, metric: str) -> float | None:
    """Calcula la métrica indicada sobre predicciones ya generadas."""
    try:
        if metric == "accuracy":
            return round(accuracy_score(y_true, y_pred), 4)
        elif metric == "balanced_accuracy":
            return round(balanced_accuracy_score(y_true, y_pred), 4)
        elif metric == "f1":
            # Usar la clase minoritaria como positiva
            pos_label = y_true.value_counts().index[-1]
            return round(f1_score(y_true, y_pred, average="binary", pos_label=pos_label), 4)
        elif metric == "f1_macro":
            return round(f1_score(y_true, y_pred, average="macro"), 4)
        elif metric == "root_mean_squared_error":
            return round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
        else:
            return round(accuracy_score(y_true, y_pred), 4)
    except Exception as e:
        print(f"      ⚠ Error evaluando {metric}: {e}")
        return None


def train_autogluon(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label: str,
    problem_type: str,
    eval_metric: str,
    preset,
    time_limit: int,
    calibrate: bool,
    save_path: str,
) -> dict:
    """
    Entrena AutoGluon y devuelve predicciones y tiempo real de entrenamiento.
    verbosity=0 para no saturar la consola durante la comparativa.
    """
    predictor = TabularPredictor(
        label=label,
        path=save_path,
        verbosity=0,
        problem_type=problem_type,
        eval_metric=eval_metric,
    )

    t0 = time.time()
    predictor.fit(
        train_data=train_data,
        presets=preset,
        time_limit=time_limit,
    )
    training_time = round(time.time() - t0, 1)

    if calibrate:
        try:
            predictor.calibrate_decision_threshold()
        except Exception:
            pass

    y_true = test_data[label]
    y_pred = predictor.predict(test_data)

    return {
        "y_true":          y_true,
        "y_pred":          y_pred,
        "training_time_s": training_time,
    }


def _fmt(value, metric: str) -> str:
    """Formatea un score para mostrar por consola."""
    if value is None:
        return "—"
    if metric == "root_mean_squared_error":
        return f"{value:.4f}"
    return f"{value * 100:.2f}%"


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results  = []
    summary_lines = []

    for entry in DATASETS:
        name       = entry["name"]
        label      = entry["label"]
        m_baseline = entry["baseline_metric"]

        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  Dataset: {name}")
        print(f"  {entry['description']}")
        print(sep)

        # --- Cargar dataset ---
        try:
            df = pd.read_csv(entry["csv_path"])
            print(f"  Filas: {len(df):,} | Columnas: {df.shape[1]}")
        except FileNotFoundError:
            print(f"  ⚠ Archivo no encontrado: {entry['csv_path']} — saltando.")
            continue

        # División train/test con semilla fija para reproducibilidad
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

        # =================================================================
        # PASO 1 — Policy engine
        # =================================================================
        print(f"\n  [POLICY ENGINE] Analizando dataset y generando configuración…")
        print(f"  Respuestas del usuario: {entry['user_answers']}")

        try:
            config = policy_engine_run(
                source=train_df,
                label=label,
                priority=entry["user_answers"]["priority"],
                time_budget_level=entry["user_answers"]["time_budget_level"],
                focus_minority_class=entry["user_answers"]["focus_minority_class"],
                deployment_needed=entry["user_answers"]["deployment_needed"],
                confirmed_problem_type=entry["confirmed_problem_type"],
            )

            m_policy    = config["predictor_init"]["eval_metric"]
            ptype       = config["predictor_init"]["problem_type"]
            preset_pe   = config["fit_args"]["presets"]
            time_lim_pe = config["fit_args"]["time_limit"]
            calibrate   = config["post_fit"]["calibrate_decision_threshold"]
            preset_pe_str = preset_pe[0] if isinstance(preset_pe, list) else preset_pe

            print(f"  → problem_type : {ptype}")
            print(f"  → eval_metric  : {m_policy}")
            print(f"  → presets      : {preset_pe}")
            print(f"  → time_limit   : {time_lim_pe}s")
            print(f"  → calibrate    : {calibrate}")
            print("  → Notas:")
            for note in config["notes"]:
                print(f"       • {note}")

        except AmbiguousProblemTypeError as e:
            print(f"  ✗ Tipo de problema ambiguo: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Error en el policy engine: {e}")
            import traceback; traceback.print_exc()
            continue

        # =================================================================
        # PASO 2 — Entrenamiento: Policy Engine
        # =================================================================
        print(f"\n  [POLICY ENGINE] Entrenando…  ({preset_pe_str} | {time_lim_pe}s | {m_policy})")
        try:
            pe_res = train_autogluon(
                train_df, test_df, label,
                problem_type=ptype,
                eval_metric=m_policy,
                preset=preset_pe,
                time_limit=time_lim_pe,
                calibrate=calibrate,
                save_path=f"./ag_policy_{name.replace(' ', '_')}",
            )
            y_true    = pe_res["y_true"]
            y_pred_pe = pe_res["y_pred"]
            pe_score_baseline = evaluate_manual(y_true, y_pred_pe, m_baseline)
            pe_score_policy   = evaluate_manual(y_true, y_pred_pe, m_policy)
            print(f"    {m_baseline:<30}: {_fmt(pe_score_baseline, m_baseline)}")
            print(f"    {m_policy:<30}: {_fmt(pe_score_policy, m_policy)}")
            print(f"    Tiempo real: {pe_res['training_time_s']}s")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback; traceback.print_exc()
            pe_res = {"training_time_s": None}
            pe_score_baseline = pe_score_policy = None
            y_true = test_df[label]

        # =================================================================
        # PASO 3 — Entrenamiento: Baseline A (300s, usuario sin experiencia)
        # =================================================================
        print(f"\n  [BASELINE A]    Entrenando…  ({BASELINE_PRESET} | {BASELINE_TIME_A}s | {m_baseline})")
        try:
            bl_a_res = train_autogluon(
                train_df, test_df, label,
                problem_type=ptype,
                eval_metric=m_baseline,
                preset=BASELINE_PRESET,
                time_limit=BASELINE_TIME_A,
                calibrate=False,
                save_path=f"./ag_baseline_a_{name.replace(' ', '_')}",
            )
            y_pred_bla = bl_a_res["y_pred"]
            bla_score_baseline = evaluate_manual(y_true, y_pred_bla, m_baseline)
            bla_score_policy   = evaluate_manual(y_true, y_pred_bla, m_policy)
            print(f"    {m_baseline:<30}: {_fmt(bla_score_baseline, m_baseline)}")
            print(f"    {m_policy:<30}: {_fmt(bla_score_policy, m_policy)}")
            print(f"    Tiempo real: {bl_a_res['training_time_s']}s")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            bl_a_res = {"training_time_s": None}
            bla_score_baseline = bla_score_policy = None

        # =================================================================
        # PASO 4 — Entrenamiento: Baseline B (mismo tiempo que policy engine)
        # Aísla el efecto de la configuración inteligente eliminando
        # la ventaja de tiempo que puede tener el policy engine.
        # =================================================================
        print(f"\n  [BASELINE B]    Entrenando…  ({BASELINE_PRESET} | {time_lim_pe}s | {m_baseline})")
        try:
            bl_b_res = train_autogluon(
                train_df, test_df, label,
                problem_type=ptype,
                eval_metric=m_baseline,
                preset=BASELINE_PRESET,
                time_limit=time_lim_pe,
                calibrate=False,
                save_path=f"./ag_baseline_b_{name.replace(' ', '_')}",
            )
            y_pred_blb = bl_b_res["y_pred"]
            blb_score_baseline = evaluate_manual(y_true, y_pred_blb, m_baseline)
            blb_score_policy   = evaluate_manual(y_true, y_pred_blb, m_policy)
            print(f"    {m_baseline:<30}: {_fmt(blb_score_baseline, m_baseline)}")
            print(f"    {m_policy:<30}: {_fmt(blb_score_policy, m_policy)}")
            print(f"    Tiempo real: {bl_b_res['training_time_s']}s")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            bl_b_res = {"training_time_s": None}
            blb_score_baseline = blb_score_policy = None

        # --- Registrar resultados ---
        all_results.append({
            "Dataset":                    name,
            "Descripcion":                entry["description"],
            "Problem_type":               ptype,
            "Metrica_baseline":           m_baseline,
            "Metrica_policy":             m_policy,
            # Policy engine
            "Policy_Preset":              str(preset_pe),
            "Policy_Tiempo_s":            pe_res["training_time_s"],
            f"Policy_{m_baseline}":       pe_score_baseline,
            f"Policy_{m_policy}":         pe_score_policy,
            # Baseline A (300s)
            "BaselineA_Preset":           BASELINE_PRESET,
            "BaselineA_Tiempo_s":         bl_a_res["training_time_s"],
            f"BaselineA_{m_baseline}":    bla_score_baseline,
            f"BaselineA_{m_policy}":      bla_score_policy,
            # Baseline B (mismo tiempo)
            "BaselineB_Preset":           BASELINE_PRESET,
            "BaselineB_Tiempo_s":         bl_b_res["training_time_s"],
            f"BaselineB_{m_baseline}":    blb_score_baseline,
            f"BaselineB_{m_policy}":      blb_score_policy,
        })

        # Línea de resumen para el .txt
        def _delta(pe, bl):
            if pe is None or bl is None:
                return "—"
            d = pe - bl
            sign = "+" if d >= 0 else ""
            if m_policy == "root_mean_squared_error":
                # Para error: negativo es mejor
                sign = "+" if d <= 0 else ""
                return f"{sign}{abs(d)*100:.2f}pp mejor" if d <= 0 else f"+{d*100:.2f}pp peor"
            return f"{sign}{d*100:.2f}pp"

        summary_lines.append(
            f"\n{name} ({m_policy})\n"
            f"  Policy Engine : {_fmt(pe_score_policy, m_policy)}\n"
            f"  Baseline A    : {_fmt(bla_score_policy, m_policy)}  "
            f"(delta vs policy: {_delta(pe_score_policy, bla_score_policy)})\n"
            f"  Baseline B    : {_fmt(blb_score_policy, m_policy)}  "
            f"(delta vs policy: {_delta(pe_score_policy, blb_score_policy)})"
        )

    # =============================================================================
    # GUARDAR RESULTADOS
    # =============================================================================
    if not all_results:
        print("\n⚠ No se completó ningún dataset. Revisa las rutas de los CSV.")
        return

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("comparativa_resultados.csv", index=False, encoding="utf-8-sig")

    resumen_text = (
        "COMPARATIVA POLICY ENGINE vs BASELINE\n"
        "=" * 65 + "\n"
        "Baseline A: medium_quality, 300s, sin configuración inteligente\n"
        "Baseline B: medium_quality, mismo tiempo que policy engine\n"
        "Policy:     configuración automática del motor de decisión\n"
        "=" * 65
        + "\n".join(summary_lines)
        + f"\n\n{'='*65}\n"
        "Resultados completos en comparativa_resultados.csv\n"
    )

    with open("comparativa_resumen.txt", "w", encoding="utf-8") as f:
        f.write(resumen_text)

    print(f"\n\n{'='*65}")
    print(resumen_text)
    print("✓ Guardado: comparativa_resultados.csv")
    print("✓ Guardado: comparativa_resumen.txt")


if __name__ == "__main__":
    main()