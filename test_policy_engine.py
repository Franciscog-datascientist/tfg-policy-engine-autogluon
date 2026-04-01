"""
test_policy_engine.py — Tests unitarios del motor de decisión (V1)

Cubre las cinco reglas de decisión del policy engine:
    1. problem_type  (7 tests)
    2. eval_metric   (8 tests)
    3. presets       (9 tests)
    4. time_limit    (3 tests)
    5. calibrate_decision_threshold (4 tests)
    6. validación de entradas       (4 tests)
    7. integración end-to-end       (4 tests)

Total: 39 tests

Ejecución:
    py -3.11 -m pytest test_policy_engine.py -v
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_engine import (
    _decide_problem_type,
    _decide_eval_metric,
    _decide_presets,
    _decide_time_limit,
    _decide_calibrate_threshold,
    _validate_user_goals,
    AmbiguousProblemTypeError,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_meta(dtype="categorical", n_unique=2, imbalance_ratio=1.0):
    return {
        "target_dtype": dtype,
        "target_n_unique": n_unique,
        "imbalance_ratio": imbalance_ratio,
    }


def _make_binary_df(n=200, imbalance=False):
    """DataFrame mínimo con target binario."""
    if imbalance:
        target = [0] * 170 + [1] * 30
    else:
        target = [0, 1] * (n // 2)
    return pd.DataFrame({
        "feature1": range(n),
        "feature2": [float(i) for i in range(n)],
        "target": target[:n],
    })


def _make_regression_df(n=200):
    """DataFrame mínimo con target de regresión (muchos únicos numéricos)."""
    import random
    random.seed(42)
    return pd.DataFrame({
        "feature1": range(n),
        "target": [round(i * 0.37 + random.random(), 4) for i in range(n)],
    })


def _make_multiclass_df(n=300):
    """DataFrame mínimo con target multiclase categórico."""
    classes = ["A", "B", "C"] * (n // 3)
    return pd.DataFrame({
        "feature1": range(n),
        "target": classes[:n],
    })


# ---------------------------------------------------------------------------
# 1. problem_type (7 tests)
# ---------------------------------------------------------------------------

class TestDecideProblemType:

    def test_binary_two_unique_values(self):
        meta = _target_meta(dtype="categorical", n_unique=2)
        notes = []
        result = _decide_problem_type(meta, n_rows=500, notes=notes)
        assert result == "binary"

    def test_binary_boolean_two_unique(self):
        meta = _target_meta(dtype="boolean", n_unique=2)
        notes = []
        result = _decide_problem_type(meta, n_rows=500, notes=notes)
        assert result == "binary"

    def test_multiclass_categorical_more_than_two(self):
        meta = _target_meta(dtype="categorical", n_unique=5)
        notes = []
        result = _decide_problem_type(meta, n_rows=500, notes=notes)
        assert result == "multiclass"

    def test_multiclass_boolean_more_than_two(self):
        meta = _target_meta(dtype="boolean", n_unique=4)
        notes = []
        result = _decide_problem_type(meta, n_rows=500, notes=notes)
        assert result == "multiclass"

    def test_regression_numeric_high_unique_ratio(self):
        # 100 únicos sobre 500 filas → ratio 0.20 > 0.05
        meta = _target_meta(dtype="numeric", n_unique=100)
        notes = []
        result = _decide_problem_type(meta, n_rows=500, notes=notes)
        assert result == "regression"

    def test_ambiguous_numeric_low_unique_ratio_raises(self):
        # 5 únicos sobre 500 filas → ratio 0.01 ≤ 0.05 → ambiguo
        meta = _target_meta(dtype="numeric", n_unique=5)
        notes = []
        with pytest.raises(AmbiguousProblemTypeError):
            _decide_problem_type(meta, n_rows=500, notes=notes)

    def test_confirmed_problem_type_overrides_rules(self):
        # Aunque la heurística diría binary, el usuario fuerza multiclass
        meta = _target_meta(dtype="categorical", n_unique=2)
        notes = []
        result = _decide_problem_type(
            meta, n_rows=500, notes=notes,
            confirmed_problem_type="multiclass"
        )
        assert result == "multiclass"


# ---------------------------------------------------------------------------
# 2. eval_metric (8 tests)
# ---------------------------------------------------------------------------

class TestDecideEvalMetric:

    def test_binary_focus_minority_returns_f1(self):
        meta = _target_meta(imbalance_ratio=3.0)
        notes = []
        result = _decide_eval_metric("binary", meta, "yes", notes)
        assert result == "f1"

    def test_binary_no_focus_balanced_returns_balanced_accuracy(self):
        meta = _target_meta(imbalance_ratio=2.0)
        notes = []
        result = _decide_eval_metric("binary", meta, "no", notes)
        assert result == "balanced_accuracy"

    def test_binary_no_focus_equilibrated_returns_accuracy(self):
        meta = _target_meta(imbalance_ratio=1.2)
        notes = []
        result = _decide_eval_metric("binary", meta, "no", notes)
        assert result == "accuracy"

    def test_binary_imbalance_exactly_15_returns_accuracy(self):
        # Frontera: IR = 1.5 → equilibrado (no supera 1.5)
        meta = _target_meta(imbalance_ratio=1.5)
        notes = []
        result = _decide_eval_metric("binary", meta, "no", notes)
        assert result == "accuracy"

    def test_binary_imbalance_above_15_returns_balanced_accuracy(self):
        # IR = 1.51 → desbalanceado
        meta = _target_meta(imbalance_ratio=1.51)
        notes = []
        result = _decide_eval_metric("binary", meta, "no", notes)
        assert result == "balanced_accuracy"

    def test_multiclass_imbalanced_returns_f1_macro(self):
        meta = _target_meta(imbalance_ratio=3.0)
        notes = []
        result = _decide_eval_metric("multiclass", meta, "no", notes)
        assert result == "f1_macro"

    def test_multiclass_balanced_returns_accuracy(self):
        meta = _target_meta(imbalance_ratio=1.0)
        notes = []
        result = _decide_eval_metric("multiclass", meta, "no", notes)
        assert result == "accuracy"

    def test_regression_always_returns_rmse(self):
        meta = {"target_dtype": "numeric", "target_n_unique": 100,
                "imbalance_ratio": None}
        notes = []
        result = _decide_eval_metric("regression", meta, "no", notes)
        assert result == "root_mean_squared_error"


# ---------------------------------------------------------------------------
# 3. presets (9 tests)
# ---------------------------------------------------------------------------

class TestDecidePresets:

    def test_speed_low_returns_medium_quality(self):
        notes = []
        result = _decide_presets("speed", "low", "no", 5000, notes)
        assert result == ["medium_quality"]

    def test_speed_medium_returns_medium_quality(self):
        notes = []
        result = _decide_presets("speed", "medium", "no", 5000, notes)
        assert result == ["medium_quality"]

    def test_speed_high_returns_good_quality(self):
        notes = []
        result = _decide_presets("speed", "high", "no", 5000, notes)
        assert result == ["good_quality"]

    def test_balanced_any_returns_good_quality(self):
        notes = []
        result = _decide_presets("balanced", "medium", "no", 5000, notes)
        assert result == ["good_quality"]

    def test_performance_low_returns_high_quality(self):
        notes = []
        result = _decide_presets("performance", "low", "no", 5000, notes)
        assert result == ["high_quality"]

    def test_performance_medium_returns_high_quality(self):
        notes = []
        result = _decide_presets("performance", "medium", "no", 5000, notes)
        assert result == ["high_quality"]

    def test_performance_high_returns_best_quality(self):
        notes = []
        result = _decide_presets("performance", "high", "no", 5000, notes)
        assert result == ["best_quality"]

    def test_small_dataset_forces_medium_quality(self):
        # n_rows < 1000 → siempre medium_quality, regla Hollmann et al. (2024)
        notes = []
        result = _decide_presets("performance", "high", "no", 500, notes)
        assert result == ["medium_quality"]

    def test_deployment_needed_adds_modifier(self):
        notes = []
        result = _decide_presets("balanced", "medium", "yes", 5000, notes)
        assert "optimize_for_deployment" in result


# ---------------------------------------------------------------------------
# 4. time_limit (3 tests)
# ---------------------------------------------------------------------------

class TestDecideTimeLimit:

    def test_low_returns_300(self):
        notes = []
        assert _decide_time_limit("low", notes) == 300

    def test_medium_returns_1800(self):
        notes = []
        assert _decide_time_limit("medium", notes) == 1800

    def test_high_returns_7200(self):
        notes = []
        assert _decide_time_limit("high", notes) == 7200


# ---------------------------------------------------------------------------
# 5. calibrate_decision_threshold (4 tests)
# ---------------------------------------------------------------------------

class TestDecideCalibrateThreshold:

    def test_binary_f1_returns_true(self):
        notes = []
        assert _decide_calibrate_threshold("binary", "f1", notes) is True

    def test_binary_balanced_accuracy_returns_true(self):
        notes = []
        assert _decide_calibrate_threshold("binary", "balanced_accuracy", notes) is True

    def test_binary_accuracy_returns_false(self):
        notes = []
        assert _decide_calibrate_threshold("binary", "accuracy", notes) is False

    def test_multiclass_returns_false(self):
        notes = []
        assert _decide_calibrate_threshold("multiclass", "f1_macro", notes) is False


# ---------------------------------------------------------------------------
# 6. validación de entradas (4 tests)
# ---------------------------------------------------------------------------

class TestValidateUserGoals:

    def _base_goals(self):
        return {
            "label": "target",
            "priority": "balanced",
            "time_budget_level": "medium",
            "focus_minority_class": "no",
            "deployment_needed": "no",
        }

    def test_valid_goals_do_not_raise(self):
        _validate_user_goals(self._base_goals())

    def test_invalid_priority_raises(self):
        goals = self._base_goals()
        goals["priority"] = "ultra"
        with pytest.raises(ValueError, match="priority"):
            _validate_user_goals(goals)

    def test_invalid_time_budget_raises(self):
        goals = self._base_goals()
        goals["time_budget_level"] = "extreme"
        with pytest.raises(ValueError, match="time_budget_level"):
            _validate_user_goals(goals)

    def test_missing_field_raises(self):
        goals = self._base_goals()
        del goals["deployment_needed"]
        with pytest.raises(ValueError, match="deployment_needed"):
            _validate_user_goals(goals)


# ---------------------------------------------------------------------------
# 7. integración end-to-end (4 tests)
# ---------------------------------------------------------------------------

class TestRunIntegration:

    def test_binary_balanced_produces_correct_config(self):
        df = _make_binary_df(n=2000, imbalance=False)
        config = run(df, "target", "balanced", "medium", "no", "no")
        assert config["predictor_init"]["problem_type"] == "binary"
        assert config["predictor_init"]["eval_metric"] == "accuracy"
        assert "good_quality" in config["fit_args"]["presets"]
        assert config["fit_args"]["time_limit"] == 1800
        assert config["post_fit"]["calibrate_decision_threshold"] is False

    def test_binary_imbalanced_with_focus_produces_f1_and_calibration(self):
        df = _make_binary_df(n=200, imbalance=True)
        config = run(df, "target", "balanced", "medium", "yes", "no")
        assert config["predictor_init"]["eval_metric"] == "f1"
        assert config["post_fit"]["calibrate_decision_threshold"] is True

    def test_regression_produces_rmse_and_no_calibration(self):
        df = _make_regression_df(n=200)
        config = run(df, "target", "performance", "high", "no", "no")
        assert config["predictor_init"]["problem_type"] == "regression"
        assert config["predictor_init"]["eval_metric"] == "root_mean_squared_error"
        assert config["post_fit"]["calibrate_decision_threshold"] is False

    def test_small_dataset_forces_medium_quality_preset(self):
        df = _make_binary_df(n=100, imbalance=False)
        config = run(df, "target", "performance", "high", "no", "no")
        assert config["fit_args"]["presets"] == ["medium_quality"]