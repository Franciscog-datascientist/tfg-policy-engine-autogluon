"""
dataset_analyzer.py
Módulo de análisis del dataset para el policy engine.
Recibe un DataFrame de pandas o una ruta a un archivo CSV
y devuelve el bloque dataset_meta que necesita el policy engine.
"""
 
import pandas as pd
 
 
def analyze_dataset(source: pd.DataFrame | str) -> dict:
    """
    Analiza un dataset y devuelve sus características principales.
 
    Parámetros
    ----------
    source : pd.DataFrame | str
        DataFrame de pandas ya cargado, o ruta a un archivo CSV.
 
    Retorna
    -------
    dict
        Diccionario dataset_meta con las siguientes claves:
        - n_rows: número total de filas
        - n_cols: número total de columnas
        - missing_ratio: proporción global de celdas nulas (celdas_nulas / total_celdas)
        - num_numeric_features: número de columnas numéricas
        - num_categorical_features: número de columnas categóricas u object
 
    Excepciones
    -----------
    TypeError
        Si source no es un DataFrame ni una ruta a un archivo CSV.
    FileNotFoundError
        Si la ruta proporcionada no existe.
    ValueError
        Si el dataset está vacío.
    """
 
    # --- Carga ---
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        df = pd.read_csv(source)
    else:
        raise TypeError(
            f"Se esperaba un DataFrame o una ruta (str), se recibió {type(source).__name__}."
        )
 
    # --- Validación básica ---
    if df.empty:
        raise ValueError("El dataset está vacío.")
 
    # --- Cálculos ---
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = df.isnull().sum().sum()
    missing_ratio = float(round(missing_cells / total_cells, 4))
 
    numeric_dtypes = ["int8", "int16", "int32", "int64",
                      "float16", "float32", "float64"]
    num_numeric_features = int(df.select_dtypes(include=numeric_dtypes).shape[1])
    num_categorical_features = int(df.select_dtypes(include=["object", "category", "bool"]).shape[1])
 
    dataset_meta = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_ratio": missing_ratio,
        "num_numeric_features": num_numeric_features,
        "num_categorical_features": num_categorical_features,
    }
 
    return dataset_meta
 
 
# --- Uso de ejemplo ---
if __name__ == "__main__":
    # Ejemplo con DataFrame sintético
    import numpy as np
 
    df_example = pd.DataFrame({
        "edad": [25, 30, None, 45, 22],
        "salario": [30000.0, 45000.0, 50000.0, None, 28000.0],
        "ciudad": ["Madrid", "Valencia", "Madrid", None, "Sevilla"],
        "activo": [True, False, True, True, False],
        "target": [1, 0, 1, 0, 1],
    })
 
    resultado = analyze_dataset(df_example)
    print("=== dataset_meta ===")
    for clave, valor in resultado.items():
        print(f"  {clave}: {valor}")