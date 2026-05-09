"""
Indicadores financieros para evaluacion de riesgo en PYMES.

Calcula los 18 indicadores definidos en docs/DICCIONARIO_DATOS_ML.md mas el
Z''-Score de Altman para mercados emergentes (variante defendida en el
estado del arte v2 como heuristica de etiquetado).

Las funciones operan sobre el DataFrame pivotado del dataset consolidado
(data/processed/colombia_consolidado_pymes.csv), donde cada columna es un
concepto contable de la NIIF Pymes.

Convencion: las divisiones por cero retornan NaN (no inf, no error).

Uso:
    import pandas as pd
    from src.indicadores import calcular_todos

    df = pd.read_csv('data/processed/colombia_consolidado_pymes.csv')
    indicadores = calcular_todos(df)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Mapeo de columnas SIREM -> variable interna
# ---------------------------------------------------------------------------
# Si en algun momento cambian los nombres en el CSV pivotado, ajustar aqui.

COL_ACTIVOS_TOTAL        = 'Total de activos'
COL_ACTIVOS_CORRIENTES   = 'Activos corrientes totales'
COL_INVENTARIOS          = 'Inventarios corrientes'
COL_EFECTIVO             = 'Efectivo y equivalentes al efectivo'
COL_CXC                  = 'Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes'

COL_PASIVOS_TOTAL        = 'Total pasivos'
COL_PASIVOS_CORRIENTES   = 'Pasivos corrientes totales'

COL_PATRIMONIO           = 'Patrimonio total'
COL_GANANCIAS_ACUMULADAS = 'Ganancias acumuladas'

COL_INGRESOS             = 'Ingresos de actividades ordinarias'
COL_COSTO_VENTAS         = 'Costo de ventas'
COL_GANANCIA_BRUTA       = 'Ganancia bruta'
COL_GANANCIA_OPERACIONAL = 'Ganancia (perdida) por actividades de operacion'
COL_COSTOS_FINANCIEROS   = 'Costos financieros'
COL_GANANCIA_NETA        = 'Ganancia (perdida)'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division que retorna NaN cuando el denominador es 0 o NaN."""
    den = den.where(den != 0)
    return num / den


# ---------------------------------------------------------------------------
# 1. Liquidez
# ---------------------------------------------------------------------------

def razon_corriente(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_ACTIVOS_CORRIENTES], df[COL_PASIVOS_CORRIENTES])

def prueba_acida(df: pd.DataFrame) -> pd.Series:
    inv = df[COL_INVENTARIOS].fillna(0)
    return _safe_div(df[COL_ACTIVOS_CORRIENTES] - inv, df[COL_PASIVOS_CORRIENTES])

def capital_trabajo(df: pd.DataFrame) -> pd.Series:
    return df[COL_ACTIVOS_CORRIENTES] - df[COL_PASIVOS_CORRIENTES]

def razon_efectivo(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_EFECTIVO], df[COL_PASIVOS_CORRIENTES])


# ---------------------------------------------------------------------------
# 2. Rentabilidad
# ---------------------------------------------------------------------------

def margen_bruto(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_BRUTA], df[COL_INGRESOS])

def margen_operacional(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_OPERACIONAL], df[COL_INGRESOS])

def margen_neto(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_NETA], df[COL_INGRESOS])

def roa(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_NETA], df[COL_ACTIVOS_TOTAL])

def roe(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_NETA], df[COL_PATRIMONIO])


# ---------------------------------------------------------------------------
# 3. Endeudamiento
# ---------------------------------------------------------------------------

def razon_deuda(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_PASIVOS_TOTAL], df[COL_ACTIVOS_TOTAL])

def deuda_patrimonio(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_PASIVOS_TOTAL], df[COL_PATRIMONIO])

def apalancamiento(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_ACTIVOS_TOTAL], df[COL_PATRIMONIO])

def cobertura_intereses(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_GANANCIA_OPERACIONAL], df[COL_COSTOS_FINANCIEROS])


# ---------------------------------------------------------------------------
# 4. Eficiencia / Actividad
# ---------------------------------------------------------------------------

def rotacion_activos(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_INGRESOS], df[COL_ACTIVOS_TOTAL])

def rotacion_inventarios(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_COSTO_VENTAS], df[COL_INVENTARIOS])

def rotacion_cartera(df: pd.DataFrame) -> pd.Series:
    return _safe_div(df[COL_INGRESOS], df[COL_CXC])

def dias_inventario(df: pd.DataFrame) -> pd.Series:
    base = pd.Series(365.0, index=df.index)
    return _safe_div(base, rotacion_inventarios(df))

def dias_cartera(df: pd.DataFrame) -> pd.Series:
    base = pd.Series(365.0, index=df.index)
    return _safe_div(base, rotacion_cartera(df))


# ---------------------------------------------------------------------------
# Z''-Score de Altman para mercados emergentes
# ---------------------------------------------------------------------------
# Coeficientes Altman 1995/2005 para empresas no manufactureras en mercados
# emergentes (variante recomendada por el estado del arte v2):
#
#   Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4
#
# Donde:
#   X1 = Capital de trabajo / Total activos
#   X2 = Ganancias acumuladas / Total activos
#   X3 = Ganancia operacional (EBIT) / Total activos
#   X4 = Patrimonio / Total pasivos
#
# Zonas originales:
#   Z'' > 2.6   -> Segura  (riesgo bajo)
#   1.1 < Z'' < 2.6 -> Gris    (riesgo medio)
#   Z'' < 1.1   -> Peligro (riesgo alto)
#
# El estado del arte v2 (sec. 2.3) recomienda calibrar los umbrales por
# cuartiles empiricos del dataset, no usar los originales como verdad
# absoluta. Para eso usar `clasificar_por_cuartiles()`.

Z2_UMBRAL_SEGURO = 2.6
Z2_UMBRAL_GRIS   = 1.1


def z_score_altman_emergentes(df: pd.DataFrame) -> pd.Series:
    x1 = _safe_div(capital_trabajo(df), df[COL_ACTIVOS_TOTAL])
    x2 = _safe_div(df[COL_GANANCIAS_ACUMULADAS], df[COL_ACTIVOS_TOTAL])
    x3 = _safe_div(df[COL_GANANCIA_OPERACIONAL], df[COL_ACTIVOS_TOTAL])
    x4 = _safe_div(df[COL_PATRIMONIO], df[COL_PASIVOS_TOTAL])
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def clasificar_zona_altman(z: pd.Series) -> pd.Series:
    """Etiqueta categorica usando los umbrales originales de Altman."""
    cond = [z >= Z2_UMBRAL_SEGURO, z >= Z2_UMBRAL_GRIS]
    choices = ['riesgo_bajo', 'riesgo_medio']
    return pd.Series(np.select(cond, choices, default='riesgo_alto'), index=z.index)


def clasificar_por_cuartiles(z: pd.Series) -> pd.Series:
    """
    Etiqueta riesgo segmentando el Z''-Score por terciles empiricos del dataset.
    Tercil superior -> bajo, central -> medio, inferior -> alto.

    Mas robusto que los umbrales originales cuando se aplica fuera del
    contexto de calibracion (Altman EE.UU. -> PYMES Colombia).
    """
    q = z.quantile([1/3, 2/3])
    q_low, q_high = q.iloc[0], q.iloc[1]
    cond = [z >= q_high, z >= q_low]
    choices = ['riesgo_bajo', 'riesgo_medio']
    return pd.Series(np.select(cond, choices, default='riesgo_alto'), index=z.index)


# ---------------------------------------------------------------------------
# Calculo masivo
# ---------------------------------------------------------------------------

INDICADORES = {
    'razon_corriente':       razon_corriente,
    'prueba_acida':          prueba_acida,
    'capital_trabajo':       capital_trabajo,
    'razon_efectivo':        razon_efectivo,
    'margen_bruto':          margen_bruto,
    'margen_operacional':    margen_operacional,
    'margen_neto':           margen_neto,
    'roa':                   roa,
    'roe':                   roe,
    'razon_deuda':           razon_deuda,
    'deuda_patrimonio':      deuda_patrimonio,
    'apalancamiento':        apalancamiento,
    'cobertura_intereses':   cobertura_intereses,
    'rotacion_activos':      rotacion_activos,
    'rotacion_inventarios':  rotacion_inventarios,
    'rotacion_cartera':      rotacion_cartera,
    'dias_inventario':       dias_inventario,
    'dias_cartera':          dias_cartera,
}


def calcular_todos(df: pd.DataFrame, incluir_zscore: bool = True) -> pd.DataFrame:
    """
    Calcula los 18 indicadores y opcionalmente el Z''-Score sobre el DataFrame.

    Conserva las columnas de identificacion NIT_LIMPIO y ANIO si estan presentes.
    """
    out = pd.DataFrame(index=df.index)
    for col in ('NIT_LIMPIO', 'ANIO'):
        if col in df.columns:
            out[col] = df[col]
    for nombre, func in INDICADORES.items():
        out[nombre] = func(df)
    if incluir_zscore:
        out['z_score_altman'] = z_score_altman_emergentes(df)
        out['zona_altman_original']    = clasificar_zona_altman(out['z_score_altman'])
        out['zona_altman_terciles']    = clasificar_por_cuartiles(out['z_score_altman'])
    return out
