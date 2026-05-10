"""
Genera notebooks/07_modelado.ipynb usando nbformat.

Construye el notebook de la Fase 5 del PLAN_DE_TRABAJO.md (Modelado).

Entrena 4 modelos:
  1) Logistic Regression (baseline) -- multinomial, balanced, StandardScaler
  2) Random Forest -- balanced_subsample
  3) XGBoost (modelo principal) -- sample_weight balanced, early_stopping
  4) XGBoost + SMOTE -- variante anti-desbalance via oversampling

Convenciones:
  - Imputacion (mediana) y escalado: fit SOLO en train.
  - SMOTE: aplicado SOLO sobre train (anti-leakage PLAN 5.4).
  - random_state=42, np.random.seed(42).
  - Hiperparametrizacion sobre val (sin K-Fold; el split temporal manda).

Outputs:
  - models/{logistic_regression,random_forest,xgboost,xgboost_smote}.joblib
  - reports/metrics/best_hyperparams.json
  - reports/tables/comparacion_modelos_validation.tex
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '07_modelado.ipynb'

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}
})

cells = []

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""# Fase 5 -- Modelado

**Notebook**: `07_modelado.ipynb`
**Plan**: Fase 5 (`PLAN_DE_TRABAJO.md` linea 360)
**Inputs**:
- `data/processed/nacional_train.csv` (108,522 x 77, ANIO 2016 + 2018-2021)
- `data/processed/nacional_val.csv`   (26,878 x 77, ANIO 2022)
- `data/processed/nacional_test.csv`  (50,957 x 77, ANIO 2023-2024) -- solo para sanity loading

**Outputs**:
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/xgboost_smote.joblib`
- `reports/metrics/best_hyperparams.json`
- `reports/tables/comparacion_modelos_validation.tex`

**Criterios de aceptacion** (PLAN 5.3):
- Los 4 modelos serializados se cargan correctamente con `joblib.load`.
- El mejor XGBoost (con o sin SMOTE) supera al baseline LR en F1-macro sobre val por >= 5pp.
- Ningun modelo muestra signos obvios de overfitting (gap train F1 vs val F1 < 10pp).

**Decisiones operativas**:

1. **Anti-leakage de preprocesado** (PLAN 5.4):
   - `SimpleImputer(strategy='median')`: `fit` solo con train.
   - `StandardScaler`: `fit` solo con train (LR; RF y XGB no requieren escalado).
   - SMOTE: `fit_resample` solo sobre train imputado.
2. **Hiperparametrizacion sin CV**: el split temporal es la unica particion valida
   (un K-Fold aleatorio cruzaria anyos de la misma empresa entre folds, leakage).
   Por lo tanto la "grid search" es manual sobre val: para cada combinacion,
   `fit(train)` y `evaluar(val)`; ganador = mejor F1-macro en val.
3. **XGBoost: `n_estimators` reemplazado por `early_stopping_rounds=30`**: el plan
   prescribe `n_estimators in {200, 500, 1000}`; con `early_stopping_rounds=30` y
   `cap=1000`, el entrenamiento detiene cuando `mlogloss` en val deja de mejorar
   30 rondas. Es equivalente a una busqueda fina sobre `n_estimators` y el
   `best_iteration` queda registrado por modelo.
4. **Imputacion XGBoost puro**: XGBoost maneja `NaN` nativamente, NO se imputa.
   Para `XGBoost + SMOTE` SI se imputa (SMOTE no soporta `NaN`).
5. **SMOTE**: `k_neighbors=5` (default), `random_state=42`, oversampling de
   las dos clases minoritarias (`riesgo_alto`, `riesgo_bajo`) hasta igualar
   `riesgo_medio`.
6. **Etiqueta**: encode con `LabelEncoder` ordenado alfabeticamente
   (`{0: bajo, 1: medio, 2: alto}` -- guardado en classes_ para inversion
   en Fase 6).

**Cronometraje esperado**: ~30-45 min en CPU multi-core (16 hilos), dominado
por XGBoost+SMOTE."""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""from __future__ import annotations
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Reproducibilidad (PLAN 2.2)
np.random.seed(42)
RANDOM_STATE = 42

# Rutas
ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
MODELS.mkdir(parents=True, exist_ok=True)
(REPORTS / 'metrics').mkdir(parents=True, exist_ok=True)
(REPORTS / 'tables').mkdir(parents=True, exist_ok=True)

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('joblib', joblib.__version__)
print('seed:', RANDOM_STATE)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 1. Carga de splits y separacion features / target

Cargamos los 3 CSV de Fase 4 con `NIT_LIMPIO` como `str` (preservar ceros).
Definimos las columnas de features eliminando: identificadores (`NIT_LIMPIO`,
`ANIO`), categoricas de texto (diagnostico, no entran al modelo) y la etiqueta.
"""))

cells.append(nbf.v4.new_code_cell("""def cargar_split(nombre: str) -> pd.DataFrame:
    ruta = DATA_PROCESSED / f'nacional_{nombre}.csv'
    return pd.read_csv(ruta, low_memory=False, dtype={'NIT_LIMPIO': str})


df_train = cargar_split('train')
df_val = cargar_split('val')
df_test = cargar_split('test')  # solo sanity-loading; no se usa en Fase 5

# Columnas que NO entran al modelo
COLS_ID = ['NIT_LIMPIO', 'ANIO']
COLS_CATEGORICAS_TEXTO = ['ciiu_seccion', 'sociedad', 'departamento']
COL_TARGET = 'etiqueta_final'
COLS_DROP = COLS_ID + COLS_CATEGORICAS_TEXTO + [COL_TARGET]

FEATURE_COLS = [c for c in df_train.columns if c not in COLS_DROP]

print(f'Splits cargados:')
print(f'  train: {df_train.shape}')
print(f'  val:   {df_val.shape}')
print(f'  test:  {df_test.shape}')
print(f'\\nFeatures ML: {len(FEATURE_COLS)}')
print(f'  Primeras 6: {FEATURE_COLS[:6]}')
print(f'  Ultimas 6:  {FEATURE_COLS[-6:]}')
print(f'\\nDistribucion etiqueta_final (train):')
print(df_train[COL_TARGET].value_counts(normalize=True).round(4).to_string())
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 2. Construccion de matrices X / y + LabelEncoder

`LabelEncoder` ordena alfabeticamente: `{0: riesgo_alto, 1: riesgo_bajo, 2: riesgo_medio}`.
Se persiste el mapeo para Fase 6 (`classes_`).
"""))

cells.append(nbf.v4.new_code_cell("""# Matrices
X_train = df_train[FEATURE_COLS].astype(np.float64).values
y_train_str = df_train[COL_TARGET].values
X_val = df_val[FEATURE_COLS].astype(np.float64).values
y_val_str = df_val[COL_TARGET].values

# Encoding de la etiqueta (consistente entre splits)
le = LabelEncoder()
le.fit(np.concatenate([y_train_str, y_val_str]))
y_train = le.transform(y_train_str)
y_val = le.transform(y_val_str)

print(f'X_train: {X_train.shape}  X_val: {X_val.shape}')
print(f'Mapeo LabelEncoder: {dict(zip(le.classes_, range(len(le.classes_))))}')
print(f'\\nDistribucion clases (codigo, conteo) train: '
      f'{dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'Distribucion clases (codigo, conteo) val:   '
      f'{dict(zip(*np.unique(y_val, return_counts=True)))}')
print(f'\\n% NaN promedio en X_train: {np.isnan(X_train).mean()*100:.2f}%')
print(f'% NaN promedio en X_val:   {np.isnan(X_val).mean()*100:.2f}%')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 3. Helpers de evaluacion

Funcion comun para evaluar un modelo (o pipeline) sobre `(X_train, y_train)` y
`(X_val, y_val)` y reportar las metricas clave de seleccion: F1-macro train,
F1-macro val y el gap. La acceptance criterion exige `gap < 10pp`.
"""))

cells.append(nbf.v4.new_code_cell("""def evaluar(estimator, X_tr, y_tr, X_vl, y_vl, etiqueta: str = ''):
    \"\"\"Devuelve dict con f1 macro train/val y gap (en puntos porcentuales).\"\"\"
    yhat_tr = estimator.predict(X_tr)
    yhat_vl = estimator.predict(X_vl)
    f1_tr = f1_score(y_tr, yhat_tr, average='macro')
    f1_vl = f1_score(y_vl, yhat_vl, average='macro')
    gap_pp = (f1_tr - f1_vl) * 100
    if etiqueta:
        print(f'  [{etiqueta}] F1-macro train={f1_tr:.4f} val={f1_vl:.4f} '
              f'gap={gap_pp:+.2f}pp')
    return {'f1_macro_train': float(f1_tr), 'f1_macro_val': float(f1_vl),
            'gap_pp': float(gap_pp)}


def reporte_clases(estimator, X_vl, y_vl, le_):
    yhat = estimator.predict(X_vl)
    rep = classification_report(y_vl, yhat, target_names=le_.classes_,
                                digits=4, zero_division=0, output_dict=True)
    return {clase: {'precision': float(rep[clase]['precision']),
                    'recall': float(rep[clase]['recall']),
                    'f1': float(rep[clase]['f1-score']),
                    'support': int(rep[clase]['support'])}
            for clase in le_.classes_}


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
print('Helpers listos.')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 4. Modelo 1 -- Logistic Regression (baseline)

- Pipeline: `SimpleImputer(median) -> StandardScaler -> LogisticRegression`.
- `solver='lbfgs'` (default; en sklearn 1.8 lbfgs ajusta multinomial automaticamente
  para 3 clases; `multi_class` fue removido de la API).
- `class_weight='balanced'` (sin SMOTE -- LR ya pondera).
- Grid: `C in {0.01, 0.1, 1, 10}` (PLAN 5.1).
"""))

cells.append(nbf.v4.new_code_cell("""LR_GRID = {'C': [0.01, 0.1, 1.0, 10.0]}

print('=== Logistic Regression baseline ===')
print(f'Grid: {LR_GRID}')
t0 = time.time()
lr_runs = []
for C in LR_GRID['C']:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs', class_weight='balanced',
            C=C, max_iter=2000,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    metr = evaluar(pipe, X_train, y_train, X_val, y_val, etiqueta=f'C={C}')
    lr_runs.append({'hp': {'C': C}, **metr, 'pipeline': pipe})

# Mejor por F1-macro val
lr_best = max(lr_runs, key=lambda r: r['f1_macro_val'])
print(f'\\nMejor LR: C={lr_best[\"hp\"][\"C\"]}  F1-macro val={lr_best[\"f1_macro_val\"]:.4f}  '
      f'gap={lr_best[\"gap_pp\"]:+.2f}pp')
print(f'Tiempo total LR: {time.time()-t0:.1f}s')

LR_BEST_PIPE = lr_best['pipeline']
LR_BEST_HP = lr_best['hp']
LR_BEST_METR = {k: v for k, v in lr_best.items() if k not in ('pipeline', 'hp')}
LR_BEST_REPORT = reporte_clases(LR_BEST_PIPE, X_val, y_val, le)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 5. Modelo 2 -- Random Forest

- Pipeline: `SimpleImputer(median) -> RandomForestClassifier`.
- `class_weight='balanced_subsample'`, `random_state=42`, `n_jobs=-1`.
- Grid: `n_estimators in {100, 300, 500}`, `max_depth in {None, 10, 20}`,
  `min_samples_split in {2, 10}` (PLAN 5.1).
- Total: 3 x 3 x 2 = 18 combinaciones.
"""))

cells.append(nbf.v4.new_code_cell("""RF_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
}

combinaciones_rf = list(product(*RF_GRID.values()))
print(f'=== Random Forest === ({len(combinaciones_rf)} combinaciones)')
t0 = time.time()
rf_runs = []
for i, (n_est, md, mss) in enumerate(combinaciones_rf, 1):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=n_est, max_depth=md, min_samples_split=mss,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    label = f'{i:02d}/18 n={n_est} d={md} mss={mss}'
    metr = evaluar(pipe, X_train, y_train, X_val, y_val, etiqueta=label)
    rf_runs.append({'hp': {'n_estimators': n_est, 'max_depth': md,
                            'min_samples_split': mss},
                    **metr, 'pipeline': pipe})

rf_best = max(rf_runs, key=lambda r: r['f1_macro_val'])
print(f'\\nMejor RF: {rf_best[\"hp\"]}')
print(f'  F1-macro val={rf_best[\"f1_macro_val\"]:.4f}  gap={rf_best[\"gap_pp\"]:+.2f}pp')
print(f'Tiempo total RF: {time.time()-t0:.1f}s')

RF_BEST_PIPE = rf_best['pipeline']
RF_BEST_HP = rf_best['hp']
RF_BEST_METR = {k: v for k, v in rf_best.items() if k not in ('pipeline', 'hp')}
RF_BEST_REPORT = reporte_clases(RF_BEST_PIPE, X_val, y_val, le)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 6. Modelo 3 -- XGBoost (sample_weight balanced)

- Sin pipeline: XGBoost maneja `NaN` nativamente.
- `objective='multi:softprob'`, `num_class=3`, `eval_metric='mlogloss'`,
  `tree_method='hist'`, `device='cuda'` si hay GPU disponible (RTX 2060
  detectada; XGBoost 3.2.0 con CUDA 12.9). Fallback a CPU automatico.
- Desbalance: `sample_weight = compute_sample_weight('balanced', y_train)`.
- `early_stopping_rounds=30` con `eval_set=[(X_val, y_val)]` (reemplaza la
  busqueda explicita de `n_estimators`).
- Grid: `max_depth in {4, 6, 8}`, `learning_rate in {0.05, 0.1}`,
  `subsample in {0.8, 1.0}`, `colsample_bytree in {0.8, 1.0}`.
- Total: 3 x 2 x 2 x 2 = 24 combinaciones.
"""))

cells.append(nbf.v4.new_code_cell("""sample_weight_train = compute_sample_weight('balanced', y_train)

# Deteccion de GPU para XGBoost
def _detectar_device_xgb():
    try:
        from xgboost import build_info
        info = build_info()
        if info.get('USE_CUDA'):
            import subprocess
            r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if r.returncode == 0:
                return 'cuda'
    except Exception:
        pass
    return 'cpu'


XGB_DEVICE = _detectar_device_xgb()
print(f'XGBoost device: {XGB_DEVICE}')

XGB_GRID = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
combinaciones_xgb = list(product(*XGB_GRID.values()))
print(f'=== XGBoost (sample_weight balanced) === ({len(combinaciones_xgb)} combinaciones)')
t0 = time.time()
xgb_runs = []
for i, (md, lr_, ss, cs) in enumerate(combinaciones_xgb, 1):
    model = XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=1000,
        max_depth=md, learning_rate=lr_,
        subsample=ss, colsample_bytree=cs,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        tree_method='hist',
        device=XGB_DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train,
              sample_weight=sample_weight_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    metr = evaluar(model, X_train, y_train, X_val, y_val,
                   etiqueta=f'{i:02d}/24 d={md} lr={lr_} ss={ss} cs={cs} '
                            f'best_iter={model.best_iteration}')
    xgb_runs.append({'hp': {'max_depth': md, 'learning_rate': lr_,
                             'subsample': ss, 'colsample_bytree': cs,
                             'best_iteration': int(model.best_iteration),
                             'n_estimators_cap': 1000,
                             'early_stopping_rounds': 30},
                     **metr, 'model': model})

xgb_best = max(xgb_runs, key=lambda r: r['f1_macro_val'])
print(f'\\nMejor XGB (sample_weight): {xgb_best[\"hp\"]}')
print(f'  F1-macro val={xgb_best[\"f1_macro_val\"]:.4f}  gap={xgb_best[\"gap_pp\"]:+.2f}pp')
print(f'Tiempo total XGB: {time.time()-t0:.1f}s')

XGB_BEST_MODEL = xgb_best['model']
XGB_BEST_HP = xgb_best['hp']
XGB_BEST_METR = {k: v for k, v in xgb_best.items() if k not in ('model', 'hp')}
XGB_BEST_REPORT = reporte_clases(XGB_BEST_MODEL, X_val, y_val, le)
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 7. Modelo 4 -- XGBoost + SMOTE

- SMOTE necesita datos sin `NaN`: imputamos primero con `SimpleImputer(median)`
  fiteado en train.
- SMOTE oversamplea las dos clases minoritarias hasta igualar la mayoritaria
  (`riesgo_medio`); aplicado SOLO sobre train (PLAN 5.4: anti-leakage).
- XGBoost con la misma configuracion que el modelo 3, pero **sin** `sample_weight`
  (el balance ya lo da SMOTE).
- Mismo grid (24 combinaciones).
- Persistencia: tupla `(imputer, model)` -- el `predict` necesita imputar antes.
"""))

cells.append(nbf.v4.new_code_cell("""# Imputador train -> aplicar a train y val (anti-leakage)
imputer_smote = SimpleImputer(strategy='median')
imputer_smote.fit(X_train)
X_train_imp = imputer_smote.transform(X_train)
X_val_imp = imputer_smote.transform(X_val)

# SMOTE sobre train imputado (val NO se toca)
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_smote, y_train_smote = sm.fit_resample(X_train_imp, y_train)

print(f'Pre-SMOTE train: {X_train_imp.shape}, '
      f'distribucion {dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'Post-SMOTE train: {X_train_smote.shape}, '
      f'distribucion {dict(zip(*np.unique(y_train_smote, return_counts=True)))}')

print(f'\\n=== XGBoost + SMOTE === ({len(combinaciones_xgb)} combinaciones)')
t0 = time.time()
xgb_smote_runs = []
for i, (md, lr_, ss, cs) in enumerate(combinaciones_xgb, 1):
    model = XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=1000,
        max_depth=md, learning_rate=lr_,
        subsample=ss, colsample_bytree=cs,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        tree_method='hist',
        device=XGB_DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_train_smote, y_train_smote,
              eval_set=[(X_val_imp, y_val)],
              verbose=False)
    # Predicciones SOBRE TRAIN ORIGINAL (no SMOTE) para gap honesto
    yhat_tr = model.predict(X_train_imp)
    yhat_vl = model.predict(X_val_imp)
    f1_tr = f1_score(y_train, yhat_tr, average='macro')
    f1_vl = f1_score(y_val, yhat_vl, average='macro')
    gap_pp = (f1_tr - f1_vl) * 100
    print(f'  {i:02d}/24 d={md} lr={lr_} ss={ss} cs={cs} '
          f'best_iter={model.best_iteration}  F1-macro train={f1_tr:.4f} val={f1_vl:.4f} '
          f'gap={gap_pp:+.2f}pp')
    xgb_smote_runs.append({
        'hp': {'max_depth': md, 'learning_rate': lr_,
               'subsample': ss, 'colsample_bytree': cs,
               'best_iteration': int(model.best_iteration),
               'n_estimators_cap': 1000,
               'early_stopping_rounds': 30,
               'smote_k_neighbors': 5},
        'f1_macro_train': float(f1_tr),
        'f1_macro_val': float(f1_vl),
        'gap_pp': float(gap_pp),
        'model': model,
    })

xgb_smote_best = max(xgb_smote_runs, key=lambda r: r['f1_macro_val'])
print(f'\\nMejor XGB+SMOTE: {xgb_smote_best[\"hp\"]}')
print(f'  F1-macro val={xgb_smote_best[\"f1_macro_val\"]:.4f}  '
      f'gap={xgb_smote_best[\"gap_pp\"]:+.2f}pp')
print(f'Tiempo total XGB+SMOTE: {time.time()-t0:.1f}s')

XGB_SMOTE_BEST_MODEL = xgb_smote_best['model']
XGB_SMOTE_BEST_HP = xgb_smote_best['hp']
XGB_SMOTE_BEST_METR = {k: v for k, v in xgb_smote_best.items() if k not in ('model', 'hp')}
# Reporte por clase usando X_val_imp
from sklearn.metrics import classification_report as _cr
_yhat_smote = XGB_SMOTE_BEST_MODEL.predict(X_val_imp)
_rep = _cr(y_val, _yhat_smote, target_names=le.classes_, digits=4,
           zero_division=0, output_dict=True)
XGB_SMOTE_BEST_REPORT = {
    clase: {'precision': float(_rep[clase]['precision']),
            'recall': float(_rep[clase]['recall']),
            'f1': float(_rep[clase]['f1-score']),
            'support': int(_rep[clase]['support'])}
    for clase in le.classes_
}
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 8. Comparacion de los 4 modelos sobre validation

Construimos un DataFrame resumen y guardamos:
- `reports/tables/comparacion_modelos_validation.tex` (tabla LaTeX para informe).
- `reports/metrics/best_hyperparams.json` (HP ganadores + metricas).
"""))

cells.append(nbf.v4.new_code_cell("""resumen_modelos = pd.DataFrame([
    {'modelo': 'LogisticRegression',
     'f1_macro_train': LR_BEST_METR['f1_macro_train'],
     'f1_macro_val': LR_BEST_METR['f1_macro_val'],
     'gap_pp': LR_BEST_METR['gap_pp'],
     'hp': LR_BEST_HP},
    {'modelo': 'RandomForest',
     'f1_macro_train': RF_BEST_METR['f1_macro_train'],
     'f1_macro_val': RF_BEST_METR['f1_macro_val'],
     'gap_pp': RF_BEST_METR['gap_pp'],
     'hp': RF_BEST_HP},
    {'modelo': 'XGBoost',
     'f1_macro_train': XGB_BEST_METR['f1_macro_train'],
     'f1_macro_val': XGB_BEST_METR['f1_macro_val'],
     'gap_pp': XGB_BEST_METR['gap_pp'],
     'hp': XGB_BEST_HP},
    {'modelo': 'XGBoost+SMOTE',
     'f1_macro_train': XGB_SMOTE_BEST_METR['f1_macro_train'],
     'f1_macro_val': XGB_SMOTE_BEST_METR['f1_macro_val'],
     'gap_pp': XGB_SMOTE_BEST_METR['gap_pp'],
     'hp': XGB_SMOTE_BEST_HP},
])
resumen_modelos = resumen_modelos.sort_values('f1_macro_val', ascending=False).reset_index(drop=True)
print('Comparacion de los 4 mejores modelos sobre validation (orden: F1 val):')
print(resumen_modelos[['modelo', 'f1_macro_train', 'f1_macro_val', 'gap_pp']].to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""# Tabla LaTeX
def fmt(x):
    return f'{x:.4f}'


def fmt_pp(x):
    sign = '+' if x >= 0 else ''
    return f'{sign}{x:.2f}'


lineas = []
for _, row in resumen_modelos.iterrows():
    lineas.append(f'{row[\"modelo\"]} & {fmt(row[\"f1_macro_train\"])} & '
                  f'{fmt(row[\"f1_macro_val\"])} & {fmt_pp(row[\"gap_pp\"])} \\\\\\\\')

# Mejor LR para el delta
f1_val_lr = float(resumen_modelos.loc[resumen_modelos['modelo'] == 'LogisticRegression',
                                       'f1_macro_val'].iloc[0])
mejor_xgb = max(XGB_BEST_METR['f1_macro_val'], XGB_SMOTE_BEST_METR['f1_macro_val'])
delta_xgb_lr_pp = (mejor_xgb - f1_val_lr) * 100

latex = (r'''\\begin{table}[ht]
\\centering
\\small
\\caption{Comparacion de los cuatro modelos sobre el conjunto de validacion (2022).
F1-macro en train y val + gap en puntos porcentuales (PLAN 5.3 exige gap < 10pp).
Mejor XGBoost vs LR baseline en val: ''' + fmt_pp(delta_xgb_lr_pp) + r'''pp.}
\\label{tab:comparacion_modelos_validation}
\\begin{tabular}{l rrr}
\\toprule
Modelo & F1 train & F1 val & gap (pp) \\\\
\\midrule
''' + '\\n'.join(lineas) + r'''
\\bottomrule
\\end{tabular}
\\end{table}
''')

ruta_tex = REPORTS / 'tables' / 'comparacion_modelos_validation.tex'
ruta_tex.write_text(latex, encoding='utf-8')
print(f'Tabla LaTeX: {ruta_tex.relative_to(ROOT)} ({ruta_tex.stat().st_size} bytes)')
print()
print(latex)
"""))

cells.append(nbf.v4.new_code_cell("""# JSON con hiperparametros ganadores y metricas detalladas
best_hp = {
    'fase': 5,
    'random_state': RANDOM_STATE,
    'feature_count': len(FEATURE_COLS),
    'feature_cols': FEATURE_COLS,
    'label_encoder_classes': le.classes_.tolist(),
    'criterio_seleccion': 'F1-macro maximo sobre validation set (2022)',
    'modelos': {
        'logistic_regression': {
            'tipo': 'sklearn.linear_model.LogisticRegression',
            'pipeline': 'SimpleImputer(median) -> StandardScaler -> LR',
            'parametros_fijos': {
                'multi_class': 'multinomial', 'solver': 'lbfgs',
                'class_weight': 'balanced', 'max_iter': 2000,
            },
            'grid_explorado': LR_GRID,
            'mejor_hp': LR_BEST_HP,
            'metricas_val': LR_BEST_METR,
            'reporte_por_clase': LR_BEST_REPORT,
        },
        'random_forest': {
            'tipo': 'sklearn.ensemble.RandomForestClassifier',
            'pipeline': 'SimpleImputer(median) -> RF',
            'parametros_fijos': {'class_weight': 'balanced_subsample',
                                  'random_state': RANDOM_STATE, 'n_jobs': -1},
            'grid_explorado': RF_GRID,
            'mejor_hp': RF_BEST_HP,
            'metricas_val': RF_BEST_METR,
            'reporte_por_clase': RF_BEST_REPORT,
        },
        'xgboost': {
            'tipo': 'xgboost.XGBClassifier',
            'pipeline': 'XGB sobre features crudos (NaN nativo) + sample_weight balanced',
            'parametros_fijos': {
                'objective': 'multi:softprob', 'num_class': 3,
                'eval_metric': 'mlogloss', 'tree_method': 'hist',
                'early_stopping_rounds': 30, 'n_estimators_cap': 1000,
            },
            'grid_explorado': XGB_GRID,
            'mejor_hp': XGB_BEST_HP,
            'metricas_val': XGB_BEST_METR,
            'reporte_por_clase': XGB_BEST_REPORT,
        },
        'xgboost_smote': {
            'tipo': 'xgboost.XGBClassifier + imblearn.SMOTE',
            'pipeline': 'SimpleImputer(median) -> SMOTE(k=5) -> XGB (sin sample_weight)',
            'parametros_fijos': {
                'objective': 'multi:softprob', 'num_class': 3,
                'eval_metric': 'mlogloss', 'tree_method': 'hist',
                'early_stopping_rounds': 30, 'n_estimators_cap': 1000,
                'smote_k_neighbors': 5,
            },
            'grid_explorado': XGB_GRID,
            'mejor_hp': XGB_SMOTE_BEST_HP,
            'metricas_val': XGB_SMOTE_BEST_METR,
            'reporte_por_clase': XGB_SMOTE_BEST_REPORT,
        },
    },
    'criterios_aceptacion_plan': {
        'mejor_xgb_supera_LR_en_pp': float(delta_xgb_lr_pp),
        'mejor_xgb_supera_LR_ge_5pp': bool(delta_xgb_lr_pp >= 5.0),
        'gap_max_modelos': float(resumen_modelos['gap_pp'].abs().max()),
        'gap_max_lt_10pp': bool(resumen_modelos['gap_pp'].abs().max() < 10.0),
    },
}

ruta_json = REPORTS / 'metrics' / 'best_hyperparams.json'
ruta_json.write_text(json.dumps(best_hp, indent=2, ensure_ascii=False),
                     encoding='utf-8')
print(f'JSON: {ruta_json.relative_to(ROOT)} ({ruta_json.stat().st_size:,} bytes)')
print(f'\\nDelta XGB-LR en val: {delta_xgb_lr_pp:+.2f}pp '
      f'({\"OK >=5pp\" if delta_xgb_lr_pp>=5 else \"NO CUMPLE\"})')
print(f'Gap max (abs): {resumen_modelos[\"gap_pp\"].abs().max():.2f}pp '
      f'({\"OK <10pp\" if resumen_modelos[\"gap_pp\"].abs().max()<10 else \"NO CUMPLE\"})')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 9. Persistencia con joblib

Guardamos los 4 modelos en `models/`:
- `logistic_regression.joblib`: sklearn `Pipeline` (imputer+scaler+LR).
- `random_forest.joblib`: sklearn `Pipeline` (imputer+RF).
- `xgboost.joblib`: `XGBClassifier` puro.
- `xgboost_smote.joblib`: `dict {'imputer': SimpleImputer, 'model': XGBClassifier}`
  -- el predict requiere `model.predict(imputer.transform(X))`.

Adicional: meta-info en cada artefacto (`feature_cols`, `classes_`).
"""))

cells.append(nbf.v4.new_code_cell("""# Metainformacion comun
META = {
    'feature_cols': FEATURE_COLS,
    'label_encoder_classes': le.classes_.tolist(),
    'random_state': RANDOM_STATE,
}

# 1. Logistic Regression -- Pipeline auto-contenido
joblib.dump({'pipeline': LR_BEST_PIPE, 'meta': META, 'hp': LR_BEST_HP},
            MODELS / 'logistic_regression.joblib')

# 2. Random Forest -- Pipeline auto-contenido
joblib.dump({'pipeline': RF_BEST_PIPE, 'meta': META, 'hp': RF_BEST_HP},
            MODELS / 'random_forest.joblib')

# 3. XGBoost puro -- modelo + meta
joblib.dump({'model': XGB_BEST_MODEL, 'meta': META, 'hp': XGB_BEST_HP},
            MODELS / 'xgboost.joblib')

# 4. XGBoost + SMOTE -- imputer + modelo
joblib.dump({'imputer': imputer_smote, 'model': XGB_SMOTE_BEST_MODEL,
             'meta': META, 'hp': XGB_SMOTE_BEST_HP},
            MODELS / 'xgboost_smote.joblib')

for nombre in ['logistic_regression', 'random_forest', 'xgboost', 'xgboost_smote']:
    p = MODELS / f'{nombre}.joblib'
    print(f'  {nombre}.joblib -> {p.stat().st_size/1024:,.1f} KB')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 10. Verificacion criterios PLAN 5.3

Tres asserts:
1. Los 4 archivos `joblib` existen y se cargan correctamente.
2. El mejor XGBoost (puro o SMOTE) supera al baseline LR en val por >= 5pp.
3. Ningun modelo tiene gap |train-val| > 10pp.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Carga roundtrip
print('--- Verificacion 1: carga joblib ---')
artefactos = {}
for nombre in ['logistic_regression', 'random_forest', 'xgboost', 'xgboost_smote']:
    obj = joblib.load(MODELS / f'{nombre}.joblib')
    artefactos[nombre] = obj
    print(f'  OK {nombre}: keys={sorted(obj.keys())}')

# Sanity: cada uno predice sobre val correctamente
print('\\n--- Verificacion sanity: predict sobre 5 filas de val ---')
preds = {}
preds['logistic_regression'] = artefactos['logistic_regression']['pipeline'].predict(X_val[:5])
preds['random_forest'] = artefactos['random_forest']['pipeline'].predict(X_val[:5])
preds['xgboost'] = artefactos['xgboost']['model'].predict(X_val[:5])
xs_imp = artefactos['xgboost_smote']['imputer'].transform(X_val[:5])
preds['xgboost_smote'] = artefactos['xgboost_smote']['model'].predict(xs_imp)
for nombre, p in preds.items():
    print(f'  {nombre}: {p.tolist()}  (clases real: {y_val[:5].tolist()})')

# 2. Delta XGB - LR
print('\\n--- Verificacion 2: XGB supera LR por >= 5pp en F1 val ---')
print(f'  F1-macro val LR             = {LR_BEST_METR[\"f1_macro_val\"]:.4f}')
print(f'  F1-macro val XGB (sw)       = {XGB_BEST_METR[\"f1_macro_val\"]:.4f}')
print(f'  F1-macro val XGB+SMOTE      = {XGB_SMOTE_BEST_METR[\"f1_macro_val\"]:.4f}')
print(f'  Mejor XGB - LR              = {delta_xgb_lr_pp:+.2f}pp '
      f'({\"OK >=5pp\" if delta_xgb_lr_pp>=5 else \"NO CUMPLE\"})')
assert delta_xgb_lr_pp >= 5.0, (
    f'Acceptance criterion fallido: mejor XGB ({mejor_xgb:.4f}) - LR '
    f'({f1_val_lr:.4f}) = {delta_xgb_lr_pp:.2f}pp < 5pp')

# 3. Gap < 10pp
print('\\n--- Verificacion 3: gap |train-val| < 10pp en cada modelo ---')
for _, row in resumen_modelos.iterrows():
    estado = 'OK' if abs(row['gap_pp']) < 10 else 'NO CUMPLE'
    print(f'  {row[\"modelo\"]:<22} gap={row[\"gap_pp\"]:+.2f}pp ({estado})')
assert resumen_modelos['gap_pp'].abs().max() < 10.0, (
    f'Acceptance criterion fallido: gap max = '
    f'{resumen_modelos[\"gap_pp\"].abs().max():.2f}pp >= 10pp')

print('\\nOK Todos los criterios PLAN 5.3 cumplidos.')
"""))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("""## 11. Resumen Fase 5

| Item | Valor |
|---|---|
| Modelos entrenados | 4 (LR, RF, XGB, XGB+SMOTE) |
| Combinaciones evaluadas | LR=4, RF=18, XGB=24, XGB+SMOTE=24 -- total 70 |
| Mejor modelo en val | (ver tabla `comparacion_modelos_validation`) |
| Delta XGB - LR (val) | (ver JSON `best_hyperparams`) |
| Gap max (train-val) | (ver JSON `best_hyperparams`) |

**Salidas**:
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/xgboost_smote.joblib`
- `reports/metrics/best_hyperparams.json`
- `reports/tables/comparacion_modelos_validation.tex`

**Para Fase 6**:
- Cargar el mejor modelo y evaluar sobre `nacional_test.csv` (no usado aqui).
- Generar matrices de confusion, curvas ROC/PR, AUC.
- TreeSHAP global (beeswarm) y local (waterfall x9) sobre el ganador.
"""))

# ---------------------------------------------------------------------------
nb.cells = cells
NB_PATH.write_text(nbf.writes(nb), encoding='utf-8')
print(f'Notebook escrito: {NB_PATH}')
print(f'Celdas: {len(cells)}')
