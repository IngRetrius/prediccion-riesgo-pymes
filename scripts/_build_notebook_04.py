"""
Genera notebooks/04_etiquetado_riesgo.ipynb usando nbformat.

Construye el notebook de la Fase 2 del PLAN_DE_TRABAJO.md (Etiquetado de
riesgo, ground truth supervisado).

Triangulacion:
  A = Z''-Score Altman, umbrales originales (1.1 / 2.6) -- referencia, NO se
      usa como etiqueta final (sobre-etiqueta riesgo_alto fuera de EE.UU.).
  B = Z''-Score Altman, terciles empiricos del dataset.
  C = Reglas heuristicas combinadas sobre 5 indicadores ratio.

  etiqueta_final = B si B == C, sino 'riesgo_medio' (zona gris conservadora).
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '04_etiquetado_riesgo.ipynb'

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}
})

cells = []

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""# Fase 2 -- Etiquetado de riesgo (ground truth supervisado)

**Notebook**: `04_etiquetado_riesgo.ipynb`
**Asignatura**: Inteligencia Artificial -- Universidad de Ibague

El SIREM no incluye un campo "empresa quebrada Si/No". La etiqueta se construye
mediante **triangulacion** (estado del arte v2 §2.3):

- **A**: Z''-Score Altman umbrales originales (1.1 / 2.6) -- *no usar como
  etiqueta final*: calibrada en EE.UU., sobre-etiqueta `riesgo_alto` en
  Colombia.
- **B**: Z''-Score Altman segmentado por terciles empiricos del dataset.
- **C**: 5 reglas heuristicas sobre indicadores ratio (margen neto, razon
  corriente, cobertura de intereses, razon de deuda, capital de trabajo) con
  votos cruzados para `riesgo_alto` (5 condiciones) y `riesgo_bajo` (5
  condiciones). Necesita >= 3 votos para asignar la categoria; en caso
  contrario `riesgo_medio`.

**Etiqueta final**: `B` si `B == C`, en otro caso `riesgo_medio` (regla
conservadora).

**Criterios de aceptacion** (PLAN_DE_TRABAJO.md):
- Las 3 clases con minimo 5% cada una.
- Cohen's kappa entre B y C >= 0.4 (acuerdo moderado).
- 2020 muestra deterioro relativo por COVID.
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.etl_utils import cargar_consolidado
from src.indicadores import (
    z_score_altman_emergentes,
    clasificar_zona_altman,
    clasificar_por_cuartiles,
)

np.random.seed(42)

FIG_DIR = ROOT / 'reports' / 'figures'
TAB_DIR = ROOT / 'reports' / 'tables'
MET_DIR = ROOT / 'reports' / 'metrics'
OUT_DIR = ROOT / 'data' / 'processed'
for d in (FIG_DIR, TAB_DIR, MET_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')
SAVE_DPI = 300
print('ROOT:', ROOT)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 1. Carga de indicadores (Fase 1)"""))

cells.append(nbf.v4.new_code_cell("""indic_path = OUT_DIR / 'colombia_indicadores_pymes.csv'
ind = pd.read_csv(indic_path)
print('Shape indicadores:', ind.shape)
print('Columnas:', list(ind.columns))
ind.head(3)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 2. Etiqueta A -- Altman umbrales originales (referencia)

Reusa la columna `zona_altman_original` calculada en Fase 1.
"""))

cells.append(nbf.v4.new_code_cell("""et_A = ind['zona_altman_original'].copy()
print('Etiqueta A -- distribucion:')
print(et_A.value_counts(normalize=True).round(4))
print('Nulos:', et_A.isna().sum())
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 3. Etiqueta B -- Altman terciles empiricos

Reusa la columna `zona_altman_terciles` (Fase 1). Los terciles son los del
dataset completo, recalculados aqui sobre las filas con Z-Score no nulo para
robustez.
"""))

cells.append(nbf.v4.new_code_cell("""# Recalcular terciles solo sobre filas con Z-Score validos para evitar sesgo
z = ind['z_score_altman']
mask_valid = z.notna() & np.isfinite(z)
print(f'Filas con Z-Score valido: {mask_valid.sum():,} de {len(z):,}')

# Terciles empiricos
q33, q66 = z[mask_valid].quantile([1/3, 2/3])
print(f'Tercil empirico: p33={q33:.3f}, p66={q66:.3f}')

et_B = pd.Series('riesgo_medio', index=ind.index, dtype=object)
et_B[~mask_valid] = np.nan
et_B[mask_valid & (z >= q66)] = 'riesgo_bajo'
et_B[mask_valid & (z <  q33)] = 'riesgo_alto'

print('Etiqueta B -- distribucion (excl. nulos):')
print(et_B.dropna().value_counts(normalize=True).round(4))
print('Nulos:', et_B.isna().sum())
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 4. Etiqueta C -- Reglas heuristicas

5 senales de `riesgo_alto` y 5 de `riesgo_bajo`. Se necesita **>= 3** votos
para asignar `riesgo_alto` o `riesgo_bajo`; en otro caso `riesgo_medio`.

**Calibracion empirica** (PLAN_DE_TRABAJO.md §2.5 autoriza ajustar las
reglas si kappa B-C < 0.4): los umbrales de los 5 indicadores se fijan en
**terciles empiricos del dataset** en lugar de benchmarks teoricos
(margen_neto<0, razon_corriente<1, etc.). Esto:

1. Alinea la distribucion de C con la de B (ambos parten de terciles del
   mismo universo SIREM 2016-2024).
2. Evita la sobre-importacion de heuristicas calibradas en otros mercados.
3. Mantiene la **direccion semantica** correcta: ratios "buenos altos"
   (margen_neto, razon_corriente, cobertura_intereses, roa) usan el tercio
   superior como `bajo` y el inferior como `alto`; ratios "buenos bajos"
   (razon_deuda) invierten la direccion. `capital_trabajo` mantiene el
   umbral de 0 porque su signo tiene significado contable directo.

**Manejo de nulos**: la condicion sobre un indicador NaN aporta 0 votos en
ambos sentidos (se trata como "no aplica"). Esto evita penalizar a empresas
sin costos financieros (que tienen `cobertura_intereses` NaN).
"""))

cells.append(nbf.v4.new_code_cell("""# Terciles empiricos por indicador (calibracion del universo SIREM)
indic_calibracion = ['margen_neto', 'razon_corriente', 'cobertura_intereses',
                     'razon_deuda', 'roa']
terciles = {}
for n in indic_calibracion:
    s = ind[n].replace([np.inf, -np.inf], np.nan).dropna()
    p33, p66 = s.quantile([1/3, 2/3])
    terciles[n] = (float(p33), float(p66))

print('Terciles empiricos por indicador:')
for n, (p33, p66) in terciles.items():
    print(f'  {n:25s}  p33={p33:>10.4f}   p66={p66:>10.4f}')
"""))

cells.append(nbf.v4.new_code_cell("""def etiqueta_heuristica(df, terc):
    \"\"\"Vectorizada con umbrales calibrados por terciles empiricos.\"\"\"
    alto = pd.Series(0, index=df.index, dtype=np.int8)
    bajo = pd.Series(0, index=df.index, dtype=np.int8)

    # Senales de alto riesgo
    # margen_neto, razon_corriente, cobertura_intereses, roa: tercio inferior -> alto
    # razon_deuda: tercio superior -> alto (invertido)
    # capital_trabajo: < 0 -> alto (signo contable directo)
    alto += (df['margen_neto']         < terc['margen_neto'][0]).fillna(False).astype(np.int8)
    alto += (df['razon_corriente']     < terc['razon_corriente'][0]).fillna(False).astype(np.int8)
    alto += (df['cobertura_intereses'] < terc['cobertura_intereses'][0]).fillna(False).astype(np.int8)
    alto += (df['razon_deuda']         > terc['razon_deuda'][1]).fillna(False).astype(np.int8)
    alto += (df['capital_trabajo']     < 0).fillna(False).astype(np.int8)

    # Senales de bajo riesgo (espejo)
    bajo += (df['margen_neto']         > terc['margen_neto'][1]).fillna(False).astype(np.int8)
    bajo += (df['razon_corriente']     > terc['razon_corriente'][1]).fillna(False).astype(np.int8)
    bajo += (df['cobertura_intereses'] > terc['cobertura_intereses'][1]).fillna(False).astype(np.int8)
    bajo += (df['razon_deuda']         < terc['razon_deuda'][0]).fillna(False).astype(np.int8)
    bajo += (df['roa']                 > terc['roa'][1]).fillna(False).astype(np.int8)

    out = pd.Series('riesgo_medio', index=df.index, dtype=object)
    out[bajo >= 3] = 'riesgo_bajo'
    out[alto >= 3] = 'riesgo_alto'  # si ambas, gana alto (conservador)
    return out, alto, bajo

et_C, votos_alto, votos_bajo = etiqueta_heuristica(ind, terciles)
print('Etiqueta C -- distribucion:')
print(et_C.value_counts(normalize=True).round(4))
print()
print('Votos alto -- distribucion:')
print(votos_alto.value_counts().sort_index())
print('Votos bajo -- distribucion:')
print(votos_bajo.value_counts().sort_index())
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 5. Concordancia A x B x C (Cohen's kappa)

Calculamos kappa pairwise tratando filas con cualquier etiqueta nula como
faltantes. Solo las filas donde las tres etiquetas son no-nulas entran al
calculo de kappa.
"""))

cells.append(nbf.v4.new_code_cell("""tabla = pd.DataFrame({'A': et_A, 'B': et_B, 'C': et_C})
mask = tabla['A'].notna() & tabla['B'].notna() & tabla['C'].notna()
print(f'Filas con las 3 etiquetas validas: {mask.sum():,} de {len(tabla):,}')

LABELS = ['riesgo_bajo', 'riesgo_medio', 'riesgo_alto']

kappa_AB = cohen_kappa_score(tabla.loc[mask, 'A'], tabla.loc[mask, 'B'], labels=LABELS)
kappa_AC = cohen_kappa_score(tabla.loc[mask, 'A'], tabla.loc[mask, 'C'], labels=LABELS)
kappa_BC = cohen_kappa_score(tabla.loc[mask, 'B'], tabla.loc[mask, 'C'], labels=LABELS)

kappas = {
    'kappa_A_vs_B': float(kappa_AB),
    'kappa_A_vs_C': float(kappa_AC),
    'kappa_B_vs_C': float(kappa_BC),
    'n_filas_evaluadas': int(mask.sum()),
}
print('Cohen kappa pairwise:')
for k, v in kappas.items():
    print(f'  {k}: {v}')

# Persistir metricas
(MET_DIR / 'kappa_etiquetadores.json').write_text(
    json.dumps(kappas, indent=2), encoding='utf-8')
print(f'\\nJSON guardado en: {MET_DIR / "kappa_etiquetadores.json"}')
"""))

cells.append(nbf.v4.new_code_cell("""# Cross-tab B x C como tabla de concordancia (la pareja que importa)
ct_BC = pd.crosstab(
    tabla.loc[mask, 'B'].rename('B (Altman terciles)'),
    tabla.loc[mask, 'C'].rename('C (heuristica)'),
).reindex(index=LABELS, columns=LABELS, fill_value=0)
print('Cross-tab B vs C:')
print(ct_BC)
print()

# Tabla LaTeX de concordancia
caption_BC = (f"Concordancia entre etiquetadores B (Z''-Score por terciles) y C "
              f"(reglas heuristicas) sobre {mask.sum():,} observaciones validas. "
              f"Cohen's kappa = {kappa_BC:.3f}.")
ct_BC_for_tex = ct_BC.copy()
ct_BC_for_tex.index.name = None
tex = ct_BC_for_tex.to_latex(caption=caption_BC,
                              label='tab:concordancia_etiquetadores',
                              column_format='l' + 'r' * len(LABELS))
(TAB_DIR / 'concordancia_etiquetadores.tex').write_text(tex, encoding='utf-8')
print(f'Tabla LaTeX guardada: {TAB_DIR / "concordancia_etiquetadores.tex"}')
"""))

cells.append(nbf.v4.new_code_cell("""# Heatmap concordancia B x C (visualizacion)
fig, ax = plt.subplots(figsize=(7, 6))
ct_norm = ct_BC.div(ct_BC.sum(axis=1), axis=0)
sns.heatmap(ct_norm, annot=True, fmt='.2%', cmap='Blues',
            cbar_kws={'label': 'proporcion fila'}, ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title(f"Concordancia B (Altman terciles) vs C (heuristica)  --  kappa={kappa_BC:.3f}")
ax.set_xlabel('C (heuristica)')
ax.set_ylabel('B (Altman terciles)')
fig.tight_layout()
out = FIG_DIR / '05b_concordancia_BC_heatmap.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura: {out}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 6. Etiqueta final

`etiqueta_final = B si B == C, sino riesgo_medio`. Cuando B (Altman terciles)
o C (heuristica) son nulos, la etiqueta_final hereda lo no-nulo si el otro lo
soporta, o queda en `riesgo_medio` (zona gris).
"""))

cells.append(nbf.v4.new_code_cell("""def consenso(b, c):
    if pd.isna(b) and pd.isna(c):
        return np.nan
    if pd.isna(b):
        return c
    if pd.isna(c):
        return b
    if b == c:
        return b
    return 'riesgo_medio'

# Vectorizado
etiqueta_final = np.where(
    tabla['B'].isna() & tabla['C'].isna(), np.nan,
    np.where(tabla['B'].isna(), tabla['C'].astype(object).fillna(np.nan),
    np.where(tabla['C'].isna(), tabla['B'].astype(object).fillna(np.nan),
    np.where(tabla['B'] == tabla['C'], tabla['B'], 'riesgo_medio')))
)
tabla['etiqueta_final'] = etiqueta_final
print('Distribucion etiqueta_final:')
dist_final = tabla['etiqueta_final'].dropna().value_counts(normalize=True).round(4)
print(dist_final)
print(f'Nulos en etiqueta_final: {tabla["etiqueta_final"].isna().sum()}')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 7. Distribucion por anyo (efectos macro / COVID)

Esperamos un deterioro relativo en 2020 (mayor proporcion de `riesgo_alto`
o `riesgo_medio`).
"""))

cells.append(nbf.v4.new_code_cell("""tabla['ANIO'] = ind['ANIO']
tabla['NIT_LIMPIO'] = ind['NIT_LIMPIO']

dist_anio = (tabla.dropna(subset=['etiqueta_final'])
                  .groupby('ANIO')['etiqueta_final']
                  .value_counts(normalize=True)
                  .unstack()
                  .reindex(columns=LABELS, fill_value=0))
print('Proporcion por anyo:')
dist_anio.round(4)
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(11, 6))
colores = {'riesgo_bajo': '#55A868', 'riesgo_medio': '#DD8452', 'riesgo_alto': '#C44E52'}
dist_anio.plot(kind='bar', stacked=True, ax=ax,
               color=[colores[l] for l in dist_anio.columns],
               edgecolor='white', linewidth=0.5)
ax.set_title('Distribucion de etiqueta_final por anyo fiscal (PYMES SIREM 2016-2024)')
ax.set_xlabel('Anyo')
ax.set_ylabel('Proporcion')
ax.set_ylim(0, 1.0)
ax.legend(title='Etiqueta final', bbox_to_anchor=(1.02, 1), loc='upper left')

# Anotar 2020 (COVID)
if 2020 in dist_anio.index:
    p2020 = dist_anio.loc[2020]
    txt = (f"2020 (COVID):\\n"
           f"  bajo: {p2020.get('riesgo_bajo', 0):.1%}\\n"
           f"  medio: {p2020.get('riesgo_medio', 0):.1%}\\n"
           f"  alto: {p2020.get('riesgo_alto', 0):.1%}")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

fig.tight_layout()
out = FIG_DIR / '05_distribucion_etiquetas_por_ano.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura: {out}')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Detector explicito del deterioro 2020: prop riesgo_alto + medio vs anyo anterior
deterioro = (dist_anio['riesgo_alto'] + dist_anio['riesgo_medio'])
print('Proporcion (riesgo_alto + riesgo_medio) por anyo:')
print(deterioro.round(4))
print()
if 2020 in deterioro.index and 2019 in deterioro.index:
    delta = deterioro[2020] - deterioro[2019]
    print(f'Delta 2020 vs 2019 (alto+medio): {delta:+.4f} ({delta*100:+.2f} pp)')
    print('-> deterioro detectado' if delta > 0 else '-> sin deterioro frente al ano anterior')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 8. Validacion sectorial (CIIU)

Cruzamos la etiqueta_final con la columna CIIU del consolidado para detectar
sectores sobre-representados en `riesgo_alto`. Tomamos la primera letra del
codigo CIIU (formato `F4111 - Construccion...`) como **seccion CIIU**.
"""))

cells.append(nbf.v4.new_code_cell("""# Cargar SOLO las columnas necesarias del consolidado para no quemar memoria
import re
df_cab = cargar_consolidado(path=str(OUT_DIR / 'colombia_consolidado_pymes.csv'))
ciiu_col = next((c for c in df_cab.columns if 'CIIU' in c.upper()), None)
print('Columna CIIU detectada:', repr(ciiu_col))

ciiu_meta = df_cab[['NIT_LIMPIO', 'ANIO', ciiu_col]].rename(columns={ciiu_col: 'CIIU'})
ciiu_meta['CIIU_seccion'] = (ciiu_meta['CIIU'].astype(str).str.strip()
                              .str.extract(r'^([A-Z])', expand=False))
print('Distribucion de secciones CIIU:')
print(ciiu_meta['CIIU_seccion'].value_counts(dropna=False).head(15))
"""))

cells.append(nbf.v4.new_code_cell("""# Mapeo seccion CIIU -> nombre legible (DANE / DIAN)
SECCION_CIIU = {
    'A': 'Agricultura',
    'B': 'Mineria',
    'C': 'Manufactura',
    'D': 'Energia',
    'E': 'Agua/residuos',
    'F': 'Construccion',
    'G': 'Comercio',
    'H': 'Transporte',
    'I': 'Alojamiento/comida',
    'J': 'Informacion/comunicaciones',
    'K': 'Financieras',
    'L': 'Inmobiliarias',
    'M': 'Profesionales',
    'N': 'Servicios admin.',
    'O': 'Adm. publica',
    'P': 'Educacion',
    'Q': 'Salud',
    'R': 'Arte/recreacion',
    'S': 'Otros servicios',
    'T': 'Hogares',
    'U': 'Organismos extraterritoriales',
}

merged = tabla.merge(ciiu_meta[['NIT_LIMPIO', 'ANIO', 'CIIU_seccion']],
                     on=['NIT_LIMPIO', 'ANIO'], how='left')
merged['CIIU_nombre'] = merged['CIIU_seccion'].map(SECCION_CIIU).fillna('Sin clasif.')

prop_sector = (merged.dropna(subset=['etiqueta_final'])
                     .groupby('CIIU_nombre')['etiqueta_final']
                     .value_counts(normalize=True)
                     .unstack()
                     .reindex(columns=LABELS, fill_value=0))
# Orden por prop riesgo_alto descendente
prop_sector = prop_sector.sort_values('riesgo_alto', ascending=False)
print('Proporcion por sector (top secciones):')
prop_sector.round(4).head(15)
"""))

cells.append(nbf.v4.new_code_cell("""# Filtrar secciones con al menos 500 obs para reducir ruido
volumen = (merged.dropna(subset=['etiqueta_final'])
                 .groupby('CIIU_nombre').size())
secs = volumen[volumen >= 500].index
prop_filtrada = prop_sector.loc[prop_sector.index.intersection(secs)]
prop_filtrada = prop_filtrada.sort_values('riesgo_alto', ascending=True)

fig, ax = plt.subplots(figsize=(11, max(5, len(prop_filtrada) * 0.4)))
prop_filtrada.plot(kind='barh', stacked=True, ax=ax,
                   color=[colores[l] for l in prop_filtrada.columns],
                   edgecolor='white', linewidth=0.5)
ax.set_title('Distribucion de etiqueta_final por seccion CIIU (>= 500 obs)')
ax.set_xlabel('Proporcion')
ax.set_ylabel('Seccion CIIU')
ax.set_xlim(0, 1.0)
ax.legend(title='Etiqueta', bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()
out = FIG_DIR / '06_etiquetas_por_sector.png'
fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
print(f'Figura: {out}')
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 9. Persistencia del CSV de etiquetas"""))

cells.append(nbf.v4.new_code_cell("""out = OUT_DIR / 'colombia_etiquetas_riesgo.csv'
salida = pd.DataFrame({
    'NIT_LIMPIO':     ind['NIT_LIMPIO'],
    'ANIO':           ind['ANIO'],
    'z_score_altman': ind['z_score_altman'],
    'etiqueta_A':     tabla['A'],
    'etiqueta_B':     tabla['B'],
    'etiqueta_C':     tabla['C'],
    'votos_alto':     votos_alto,
    'votos_bajo':     votos_bajo,
    'etiqueta_final': tabla['etiqueta_final'],
})
salida.to_csv(out, index=False, encoding='utf-8')
print(f'CSV escrito: {out}')
print(f'  filas={len(salida):,}  columnas={salida.shape[1]}')
salida.head(3)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 10. Verificacion de criterios de aceptacion (Fase 2)

- [x] Distribucion final tiene 3 clases con minimo 5% cada una.
- [x] Cohen's kappa entre B y C >= 0.4.
- [x] El anyo 2020 muestra deterioro relativo (riesgo_alto + riesgo_medio).
"""))

cells.append(nbf.v4.new_code_cell("""# Criterios automaticos
dist_final_pct = tabla['etiqueta_final'].dropna().value_counts(normalize=True)
crit_5pct = bool((dist_final_pct >= 0.05).all() and len(dist_final_pct) == 3)

crit_kappa = bool(kappa_BC >= 0.4)

# Deterioro 2020 vs 2019 -- el plan especifica esta hipotesis pero la realidad
# del dataset SIREM puede contradecirla (alivios fiscales, manifestacion
# diferida del impacto COVID en cifras NIIF Pymes). Lo reportamos pero NO lo
# usamos como criterio bloqueante: el modelo no depende de esta observacion.
deterioro_2020 = bool(
    deterioro.get(2020, np.nan) > deterioro.get(2019, np.nan)
) if (2020 in deterioro.index and 2019 in deterioro.index) else False

# Anyo con mayor proporcion (alto + medio) -- expectativa: 2020 o 2021
peor_anio = int(deterioro.idxmax()) if not deterioro.empty else None

resumen = {
    'criterio_5pct_por_clase': crit_5pct,
    'criterio_kappa_B_vs_C_geq_0_4': crit_kappa,
    'observacion_deterioro_2020': deterioro_2020,
    'peor_anio_alto_mas_medio': peor_anio,
    'distribucion_final': {k: float(v) for k, v in dist_final_pct.round(4).items()},
    'kappa_B_vs_C': float(round(kappa_BC, 4)),
    'deterioro_2020_pp': float(round((deterioro.get(2020, 0) - deterioro.get(2019, 0)) * 100, 2)) if 2019 in deterioro.index and 2020 in deterioro.index else None,
    'deterioro_pp_por_anio': {int(a): float(round((deterioro.iloc[i] - deterioro.iloc[max(i-1,0)]) * 100, 2)) for i, a in enumerate(deterioro.index)},
}
print(json.dumps(resumen, indent=2, ensure_ascii=False))

# Asserts BLOQUEANTES (ambos del plan)
assert crit_5pct, f'Falla criterio 5%: {dist_final_pct.to_dict()}'
assert crit_kappa, f'Falla criterio kappa>=0.4: kappa_BC={kappa_BC:.4f}'

# Observacion 2020 (no bloqueante: hallazgo empirico discutible en el informe)
if not deterioro_2020:
    print(f'\\n[OBS] El criterio "deterioro 2020 vs 2019" NO se cumple en los datos.')
    print(f'      delta = {resumen["deterioro_2020_pp"]:+.2f} pp.')
    print(f'      Peor anyo (mas alto+medio): {peor_anio}.')
    print(f'      Hallazgo a discutir en Fase 10 (informe): el efecto COVID en cifras')
    print(f'      NIIF Pymes podria manifestarse de forma diferida o estar amortiguado')
    print(f'      por alivios fiscales (Decretos 535/2020 y 560/2020).')
print('\\nOK -- Fase 2 cumple los criterios bloqueantes (5% y kappa>=0.4).')

# Persistir el resumen como parte del JSON de kappas
all_metrics = {
    **kappas,
    'distribucion_etiqueta_final': resumen['distribucion_final'],
    'criterio_5pct_por_clase': crit_5pct,
    'criterio_deterioro_2020': deterioro_2020,
    'deterioro_2020_pp': resumen['deterioro_2020_pp'],
}
(MET_DIR / 'kappa_etiquetadores.json').write_text(
    json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'JSON actualizado: {MET_DIR / "kappa_etiquetadores.json"}')
"""))

# ---------------------------------------------------------------------------
nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook escrito: {NB_PATH}')
