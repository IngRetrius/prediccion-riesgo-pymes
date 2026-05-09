"""
Utilidades para cargar el dataset consolidado y normalizar nombres de columnas.

El CSV `data/processed/colombia_consolidado_pymes.csv` tiene mojibake en
105 de sus 230 columnas: el caracter de reemplazo Unicode (U+FFFD, escrito
como `i¿½` en bytes) aparece donde originalmente habia una vocal tildada
en latin-1. Ejemplo:

    "Ganancia (pï¿½rdida)"  ->  originalmente "Ganancia (pérdida)"
    "actividades de operaciï¿½n"  ->  originalmente "actividades de operación"

Esto ocurrio porque el ETL guardo el archivo a UTF-8 despues de cargarlo
con encoding incorrecto. `normalizar_columnas()` traduce los nombres de
columna a su equivalente ASCII limpio para que `src/indicadores.py` y los
notebooks puedan referirse a las columnas con nombres estables y
portables (sin tener que escribir mojibake en el codigo).
"""
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Palabras con mojibake -> ASCII puro
# ---------------------------------------------------------------------------
# Cubre las 105 columnas afectadas del consolidado. El reemplazo es
# case-sensitive y por palabra completa (no parcial) gracias a que el
# patron mojibake es unico.

WORD_FIXES = {
    'pï¿½rdida':        'perdida',
    'pï¿½rdidas':       'perdidas',
    'operaciï¿½n':      'operacion',
    'inversiï¿½n':      'inversion',
    'disposiciï¿½n':    'disposicion',
    'participaciï¿½n':  'participacion',
    'prï¿½stamo':       'prestamo',
    'prï¿½stamos':      'prestamos',
    'biolï¿½gico':      'biologico',
    'biolï¿½gicos':     'biologicos',
    'tï¿½rmino':        'termino',
    'garantï¿½a':       'garantia',
    'mï¿½todo':         'metodo',
    'plusvalï¿½a':      'plusvalia',
    'emisiï¿½n':        'emision',
    'distribuciï¿½n':   'distribucion',
    'administraciï¿½n': 'administracion',
    'funciï¿½n':        'funcion',
    'revaluaciï¿½n':    'revaluacion',
    'disminuciï¿½n':    'disminucion',
    'variaciï¿½n':      'variacion',
    'financiaciï¿½n':   'financiacion',
    'depreciaciï¿½n':   'depreciacion',
    'amortizaciï¿½n':   'amortizacion',
    'recolocaciï¿½n':   'recolocacion',
    'readquisiciï¿½n':  'readquisicion',
}


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con nombres de columna normalizados a ASCII.

    Reemplaza tanto la version minuscula como la version con mayuscula inicial
    (e.g. tanto `inversiï¿½n` como `Inversiï¿½n`).
    """
    new_cols = {}
    for c in df.columns:
        nc = c
        for src, dst in WORD_FIXES.items():
            nc = nc.replace(src, dst)
            # variante con mayuscula inicial (al comienzo del header)
            src_cap = src[0].upper() + src[1:]
            dst_cap = dst[0].upper() + dst[1:]
            nc = nc.replace(src_cap, dst_cap)
        new_cols[c] = nc
    return df.rename(columns=new_cols)


# ---------------------------------------------------------------------------
# Carga conveniente del consolidado
# ---------------------------------------------------------------------------

DEFAULT_CONSOLIDADO = 'data/processed/colombia_consolidado_pymes.csv'


def cargar_consolidado(path: str = DEFAULT_CONSOLIDADO,
                       normalizar: bool = True,
                       **read_csv_kwargs) -> pd.DataFrame:
    """
    Carga el CSV consolidado y opcionalmente normaliza nombres de columna.

    Por defecto usa `low_memory=False` para evitar advertencias de tipo mixto.
    Cualquier kwarg adicional se pasa a `pd.read_csv` (e.g. `nrows=1000`
    para una carga rapida durante desarrollo).
    """
    kwargs = {'low_memory': False, **read_csv_kwargs}
    df = pd.read_csv(path, **kwargs)
    if normalizar:
        df = normalizar_columnas(df)
    return df


def auditar_mojibake(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de columnas que aun contienen mojibake (sanity check)."""
    return [c for c in df.columns if 'ï¿½' in c]
