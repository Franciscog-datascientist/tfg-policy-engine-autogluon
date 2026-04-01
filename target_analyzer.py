"""
target_analyzer.py — Módulo de análisis de la variable objetivo
 
Genera el bloque target_meta del policy engine a partir de un DataFrame
y el nombre de la columna objetivo (label).
 
Salidas:
    - target_dtype: tipo de dato detectado (numeric, categorical, boolean)
    - target_n_unique: número de valores únicos en la columna objetivo
    - imbalance_ratio: ratio de desbalanceo max_class / min_class (solo clasificación)
    - n_predictive_cols: número de columnas predictoras (excluyendo label)
 
Umbrales de desbalanceo (He & Garcia, 2009):
    - imbalance_ratio <= 1.5  → balanced
    - imbalance_ratio > 1.5   → slight imbalance
    - imbalance_ratio >= 10   → moderate/severe imbalance
"""
 
import pandas as pd
 
 
def _detect_target_dtype(series: pd.Series) -> str:
    """
    Detecta el tipo de dato de la variable objetivo.
 
    Retorna:
        'boolean'     — si es bool o solo tiene valores True/False/0/1
        'categorical' — si es object, category o string
        'numeric'     — si es int o float
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
 
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        return "categorical"
 
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1} or unique_vals <= {True, False}:
            return "boolean"
        return "numeric"
 
    return "categorical"
 
 
def _compute_imbalance_ratio(series: pd.Series) -> float | None:
    """
    Calcula el imbalance_ratio como max_class_count / min_class_count.
 
    Fórmula estándar propuesta por He & Garcia (2009), Wang & Yao (2012)
    y Luque et al. (2019). El resultado está acotado en [1, ∞).
 
    Retorna None si la variable tiene menos de 2 clases o si es de regresión.
    """
    value_counts = series.dropna().value_counts()
 
    if len(value_counts) < 2:
        return None
 
    max_count = value_counts.iloc[0]
    min_count = value_counts.iloc[-1]
 
    if min_count == 0:
        return None
 
    return round(max_count / min_count, 4)
 
 
def _classify_imbalance(ratio: float | None) -> str | None:
    """
    Clasifica el nivel de desbalanceo según umbrales de He & Garcia (2009).
 
    Retorna:
        'balanced'                  — ratio <= 1.5
        'slight_imbalance'          — 1.5 < ratio < 10
        'moderate_severe_imbalance' — ratio >= 10
        None                        — si ratio es None (regresión)
    """
    if ratio is None:
        return None
    if ratio <= 1.5:
        return "balanced"
    if ratio < 10:
        return "slight_imbalance"
    return "moderate_severe_imbalance"
 
 
def analyze_target(df: pd.DataFrame, label: str) -> dict:
    """
    Analiza la variable objetivo y genera el bloque target_meta.
 
    Parámetros:
        df    — DataFrame completo
        label — nombre de la columna objetivo
 
    Retorna:
        dict con las claves:
            target_dtype, target_n_unique, imbalance_ratio,
            imbalance_level, n_predictive_cols
 
    Raises:
        ValueError — si label no existe en el DataFrame
    """
    if label not in df.columns:
        raise ValueError(
            f"La columna objetivo '{label}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )
 
    target_series = df[label]
 
    target_dtype = _detect_target_dtype(target_series)
    target_n_unique = int(target_series.nunique(dropna=True))
 
    # imbalance_ratio solo se calcula para clasificación (no regresión)
    # La heurística de problem_type se aplica en el policy engine,
    # aquí calculamos el ratio siempre que haya clases discretas.
    # Para targets numéricos, usamos la heurística de AutoGluon:
    # si target_n_unique / n_rows > 0.05 → regresión (no calcular ratio).
    # Si target_n_unique <= 20, se trata como clasificación independientemente.
    if target_dtype in ("categorical", "boolean"):
        is_likely_classification = True
    elif target_n_unique == 2:
        is_likely_classification = True
    elif target_dtype == "numeric" and target_n_unique <= 20:
        is_likely_classification = True
    elif target_dtype == "numeric" and target_n_unique / len(df) <= 0.05:
        # Zona ambigua: muchos únicos pero baja proporción.
        # El policy engine pedirá confirmación al usuario.
        # Aquí NO calculamos imbalance porque no es seguro que sea clasificación.
        is_likely_classification = False
    else:
        is_likely_classification = False
 
    if is_likely_classification and target_n_unique >= 2:
        imbalance_ratio = _compute_imbalance_ratio(target_series)
    else:
        imbalance_ratio = None
 
    imbalance_level = _classify_imbalance(imbalance_ratio)
 
    n_predictive_cols = len(df.columns) - 1
 
    return {
        "target_dtype": target_dtype,
        "target_n_unique": target_n_unique,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_level": imbalance_level,
        "n_predictive_cols": n_predictive_cols,
    }
 
 
# ---------------------------------------------------------------------------
# Ejecución directa para pruebas rápidas
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 3:
        print("Uso: python target_analyzer.py <ruta_csv> <columna_objetivo>")
        print("Ejemplo: python target_analyzer.py datos.csv target")
        sys.exit(1)
 
    csv_path = sys.argv[1]
    label_col = sys.argv[2]
 
    print(f"Cargando dataset: {csv_path}")
    data = pd.read_csv(csv_path)
    print(f"Shape: {data.shape}\n")
 
    result = analyze_target(data, label_col)
 
    print("=" * 50)
    print("TARGET META")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 50)