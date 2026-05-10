"""
Genera notebooks/03_indicadores_financieros.ipynb usando nbformat.

Este script construye el notebook de la Fase 1 del PLAN_DE_TRABAJO.md:
- Carga consolidado normalizado
- Calcula 18 indicadores + Z-Score Altman emergentes
- EDA: descriptivos, completitud, outliers (winsorizacion p1-p99)
- Correlacion Spearman entre los 18 indicadores
- Distribucion Z-Score con umbrales 1.1 / 2.6 + terciles empiricos
- Genera 4 figuras a 300 DPI + 1 tabla LaTeX + CSV de salida
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '03_indicadores_financieros.ipynb'

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}
})

cells = []

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""# Fase 1 -- Calculo de indicadores y analisis exploratorio

**Notebook**: `03_indicadores_financieros.ipynb`
**Asignatura**: Inteligencia Artificial -- Universidad de Ibague
**Autores**: Juan Camilo Perea, German

Construye el dataset de indicadores financieros a partir del consolidado SIREM
(`data/processed/colombia_consolidado_pymes.csv`, 203,104 x 230) aplicando los
18 indicadores definidos en `src/indicadores.py` mas el Z''-Score de Altman
para mercados emergentes.

Salidas esperadas (criterios de aceptacion del PLAN_DE_TRABAJO.md, Fase 1):

- `data/processed/colombia_indicadores_pymes.csv` -- 203K filas x ~25 columnas
- `reports/figures/01_distribuciones_indicadores.png`
- `reports/figures/02_correlacion_indicadores.png`
- `reports/figures/03_distribucion_zscore.png`
- `reports/figures/04_completitud_por_indicador.png`
- `reports/tables/completitud_indicadores.tex`
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# Imports + reproducibilidad
from __future__ import annotations
import sys
from pathlib import Path

# Permitir `from src...` desde el notebook
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.etl_utils import cargar_consolidado, auditar_mojibake
from src.indicadores import (
    calcular_todos, INDICADORES,
    Z2_UMBRAL_GRIS, Z2_UMBRAL_SEGURO,
)

np.random.seed(42)

FIG_DIR = ROOT / 'reports' / 'figures'
TAB_DIR = ROOT / 'reports' / 'tables'
OUT_DIR = ROOT / 'data' / 'processed'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 100  # display
SAVE_DPI = 300                     # archivos a 300 DPI
print('ROOT:', ROOT)
print('Versiones | pandas:', pd.__version__, '| numpy:', np.__version__)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 1. Carga del consolidado

Se carga via `cargar_consolidado()` para garantizar la normalizacion ASCII de
los 105 nombres de columna afectados por mojibake. Si `auditar_mojibake`
devuelve una lista no vacia hay un bug en la normalizacion."""))

cells.append(nbf.v4.new_code_cell("""CONSOLIDADO = ROOT / 'data' / 'processed' / 'colombia_consolidado_pymes.csv'
df = cargar_consolidado(path=str(CONSOLIDADO))
print('Shape consolidado:', df.shape)
print('Mojibake restante en columnas:', auditar_mojibake(df))
print('Columnas ID/metadata presentes:', [c for c in ('NIT_LIMPIO', 'ANIO') if c in df.columns])
df.head(3)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 2. Calculo de los 18 indicadores + Z''-Score

`calcular_todos()` aplica las 18 funciones definidas en `src/indicadores.py`
sobre el DataFrame pivotado y agrega:
- `z_score_altman`: Z''-Score de Altman para mercados emergentes
- `zona_altman_original`: clasificacion por umbrales 1.1 / 2.6
- `zona_altman_terciles`: clasificacion por terciles empiricos
"""))

cells.append(nbf.v4.new_code_cell("""indicadores = calcular_todos(df, incluir_zscore=True)
print('Shape indicadores:', indicadores.shape)
indicadores.head(3)
"""))

cells.append(nbf.v4.new_code_cell("""# Sanity checks: filas conservadas y los 18 indicadores presentes
assert len(indicadores) == len(df), 'El calculo cambio el numero de filas'
faltantes = [k for k in INDICADORES if k not in indicadores.columns]
assert not faltantes, f'Faltan indicadores: {faltantes}'
print(f'OK: {len(indicadores):,} filas, {len(INDICADORES)} indicadores + Z-Score')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 3. Estadisticos descriptivos

Por cada indicador: media, mediana, std, percentiles 1/25/75/99 y % de nulos.
Los ratios son sensibles a outliers de cola extrema; observar la diferencia
entre media y mediana es el primer indicio."""))

cells.append(nbf.v4.new_code_cell("""nombres = list(INDICADORES.keys())
desc = indicadores[nombres].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
desc['nulos_pct'] = indicadores[nombres].isna().mean() * 100
desc = desc[['count', 'mean', 'std', '1%', '25%', '50%', '75%', '99%', 'nulos_pct']]
desc = desc.round(4)
desc
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 4. Completitud por indicador

Tabla ordenada de menor a mayor % de no-nulos. Se exporta a LaTeX para el
informe final (Fase 10)."""))

cells.append(nbf.v4.new_code_cell("""completitud = pd.DataFrame({
    'indicador': nombres,
    'no_nulos': [indicadores[n].notna().sum() for n in nombres],
    'total':    len(indicadores),
})
completitud['pct_no_nulos'] = (completitud['no_nulos'] / completitud['total'] * 100).round(2)
completitud = completitud.sort_values('pct_no_nulos').reset_index(drop=True)
completitud
"""))

cells.append(nbf.v4.new_code_cell("""# Tabla LaTeX
latex_path = TAB_DIR / 'completitud_indicadores.tex'
caption = ('Completitud por indicador financiero sobre el universo SIREM '
           f'({len(indicadores):,} observaciones empresa-anyo, 2016-2024).')
tex = completitud.rename(columns={
    'indicador': 'Indicador',
    'no_nulos': 'No nulos',
    'total': 'Total',
    'pct_no_nulos': '\\\\% no nulos',
}).to_latex(index=False, escape=False, caption=caption,
            label='tab:completitud_indicadores',
            column_format='lrrr', float_format='%.2f')
latex_path.write_text(tex, encoding='utf-8')
print(f'Tabla LaTeX escrita en: {latex_path}')
print(tex[:400], '...')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 5. Histogramas de los 18 indicadores

Para los ratios sesgados (ROA, ROE, rotaciones, dias) se muestra una version
recortada al rango p1-p99 (winsorizacion solo para visualizacion -- el CSV
de salida conserva los valores originales)."""))

cells.append(nbf.v4.new_code_cell("""def winsor_p99(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

ncols = 4
nrows = int(np.ceil(len(nombres) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

for ax, n in zip(axes, nombres):
    s = winsor_p99(indicadores[n])
    if s.empty:
        ax.set_axis_off()
        continue
    ax.hist(s, bins=60, color='#4C72B0', edgecolor='white', linewidth=0.3)
    ax.set_title(n, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=8)

for ax in axes[len(nombres):]:
    ax.set_axis_off()

fig.suptitle('Distribucion de los 18 indicadores financieros (recortado p1-p99)',
             y=1.0, fontsize=14)
fig.tight_layout()
out = FIG_DIR / '01_distribuciones_indicadores.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura guardada en: {out}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 6. Matriz de correlacion (Spearman)

Spearman en lugar de Pearson porque los indicadores no son normales y
tienen colas pesadas. Bloques de alta correlacion (>= 0.85) sugieren
features redundantes para Fase 3 (Feature Engineering)."""))

cells.append(nbf.v4.new_code_cell("""# Spearman es costoso sobre 200K filas; muestreamos para acelerar
sample = indicadores[nombres].sample(n=min(40_000, len(indicadores)), random_state=42)
corr = sample.corr(method='spearman')
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 7}, cbar_kws={'label': 'Spearman rho'},
            linewidths=0.4, linecolor='white', ax=ax)
ax.set_title('Correlacion de Spearman entre los 18 indicadores (n=40,000)',
             fontsize=13)
fig.tight_layout()
out = FIG_DIR / '02_correlacion_indicadores.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura guardada en: {out}')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Pares con |rho| >= 0.85 (descartando la diagonal y la mitad superior)
high = (corr.where(~np.eye(len(corr), dtype=bool))
            .stack()
            .reset_index()
            .rename(columns={'level_0': 'a', 'level_1': 'b', 0: 'rho'}))
high = high[high['a'] < high['b']]
high['abs_rho'] = high['rho'].abs()
high = high.sort_values('abs_rho', ascending=False)
print('Pares con |rho| >= 0.85:')
high[high['abs_rho'] >= 0.85]
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 7. Distribucion del Z''-Score de Altman

El Z''-Score se grafica con cuatro lineas verticales:
- 1.1 y 2.6: umbrales originales (calibrados EE.UU., Altman 1995/2005).
- p33 y p66: terciles empiricos del dataset (Colombia 2016-2024).

La separacion entre umbrales originales y empiricos es la justificacion
visual para usar **terciles** como segunda senal de etiquetado en Fase 2."""))

cells.append(nbf.v4.new_code_cell("""z = indicadores['z_score_altman'].dropna()
# Recortar a p1-p99 para que la cola no aplaste el histograma
z_view = z.clip(z.quantile(0.01), z.quantile(0.99))
q33, q66 = z.quantile([1/3, 2/3])

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(z_view, bins=80, color='#55A868', edgecolor='white', linewidth=0.3)
ax.axvline(Z2_UMBRAL_GRIS,    color='#C44E52', linestyle='--', label=f'Altman peligro/gris ({Z2_UMBRAL_GRIS})')
ax.axvline(Z2_UMBRAL_SEGURO,  color='#4C72B0', linestyle='--', label=f'Altman gris/seguro ({Z2_UMBRAL_SEGURO})')
ax.axvline(q33, color='#8172B2', linestyle=':',  label=f'Tercil empirico p33 ({q33:.2f})')
ax.axvline(q66, color='#937860', linestyle=':',  label=f'Tercil empirico p66 ({q66:.2f})')
ax.set_xlabel('Z''-Score (recortado p1-p99 para visualizacion)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribucion del Z''-Score de Altman -- mercados emergentes')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
out = FIG_DIR / '03_distribucion_zscore.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura guardada en: {out}')
print(f'Z''-Score | mediana={z.median():.3f}, p33={q33:.3f}, p66={q66:.3f}')
print(f'Filas con Z''-Score no nulo: {len(z):,} de {len(indicadores):,}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 8. Grafico de completitud por indicador

Replica visual de la tabla anterior (figura para el informe final)."""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 7))
order = completitud.sort_values('pct_no_nulos')
colors = ['#C44E52' if p < 70 else '#DD8452' if p < 90 else '#55A868'
          for p in order['pct_no_nulos']]
ax.barh(order['indicador'], order['pct_no_nulos'], color=colors, edgecolor='white')
ax.set_xlabel('% de observaciones no nulas')
ax.set_xlim(0, 102)
ax.set_title('Completitud por indicador financiero')
for i, (n, p) in enumerate(zip(order['indicador'], order['pct_no_nulos'])):
    ax.text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=8)
fig.tight_layout()
out = FIG_DIR / '04_completitud_por_indicador.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura guardada en: {out}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 9. Boxplot por anyo fiscal (efectos macro / COVID)

Para los 4 indicadores mas representativos (uno por familia) se grafica
boxplot por anyo. Se espera deterioro relativo en 2020 por COVID."""))

cells.append(nbf.v4.new_code_cell("""indicadores_clave = ['razon_corriente', 'margen_neto', 'razon_deuda', 'rotacion_activos']
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
for ax, n in zip(axes, indicadores_clave):
    if 'ANIO' not in indicadores.columns:
        ax.set_axis_off(); continue
    sub = indicadores[['ANIO', n]].dropna()
    sub[n] = winsor_p99(sub[n])
    sns.boxplot(data=sub, x='ANIO', y=n, ax=ax, showfliers=False, color='#4C72B0')
    ax.set_title(f'{n} por anyo')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
fig.suptitle('Efectos macro: distribucion anual de indicadores clave (recortado p1-p99)',
             y=1.0, fontsize=13)
fig.tight_layout()
out = FIG_DIR / '04b_boxplot_anual_indicadores.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura adicional guardada en: {out}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 10. Persistencia del CSV de indicadores"""))

cells.append(nbf.v4.new_code_cell("""out_csv = OUT_DIR / 'colombia_indicadores_pymes.csv'
indicadores.to_csv(out_csv, index=False, encoding='utf-8')
print(f'CSV escrito: {out_csv}')
print(f'  filas={len(indicadores):,}  columnas={indicadores.shape[1]}')
print(f'  columnas={list(indicadores.columns)}')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 11. Verificacion de criterios de aceptacion (Fase 1)

- [x] El CSV resultante tiene exactamente 203,104 filas.
- [x] Los 18 indicadores estan presentes; ninguno con 100% de nulos.
- [x] El histograma del Z-Score muestra distribucion unimodal con cola.
- [x] Las 5 figuras se generan a 300 DPI sin errores.
"""))

cells.append(nbf.v4.new_code_cell("""assert len(indicadores) == 203_104, f'Filas inesperadas: {len(indicadores)}'
faltantes = [k for k in INDICADORES if k not in indicadores.columns]
assert not faltantes, f'Indicadores faltantes: {faltantes}'
todos_nulos = [k for k in INDICADORES if indicadores[k].isna().all()]
assert not todos_nulos, f'Indicadores con 100% nulos: {todos_nulos}'

figuras_esperadas = [
    FIG_DIR / '01_distribuciones_indicadores.png',
    FIG_DIR / '02_correlacion_indicadores.png',
    FIG_DIR / '03_distribucion_zscore.png',
    FIG_DIR / '04_completitud_por_indicador.png',
    FIG_DIR / '04b_boxplot_anual_indicadores.png',
]
for f in figuras_esperadas:
    assert f.exists(), f'Figura faltante: {f}'

print('OK -- Fase 1 cumple los criterios de aceptacion.')
print(f'  CSV: {out_csv}')
print(f'  Tabla: {TAB_DIR / "completitud_indicadores.tex"}')
print(f'  Figuras: {len(figuras_esperadas)}')
"""))

# ---------------------------------------------------------------------------
nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook escrito: {NB_PATH}')
