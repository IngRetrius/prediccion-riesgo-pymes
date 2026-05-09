# Plan de Trabajo — Prediccion de Riesgo Financiero en PYMES

> **Proposito**: Este documento es el prompt operativo para ejecutar el trabajo final
> de la asignatura **Inteligencia Artificial**. Esta escrito para que un agente IA
> con acceso al repositorio pueda ejecutar el proyecto fase por fase con
> minima ambiguedad. Cada fase tiene inputs, tareas, archivos a producir y
> criterios de aceptacion verificables.

---

## 0. Resumen ejecutivo

| Item | Detalle |
|---|---|
| **Asignatura** | Inteligencia Artificial — Universidad de Ibague |
| **Autores** | Juan Camilo Perea, German |
| **Repositorio** | `IngRetrius/prediccion-riesgo-pymes` |
| **Fuente de datos** | SIREM (Supersociedades Colombia) — NIIF Pymes Grupo 2, 2016–2024 |
| **Universo** | 38,245 PYMES nacionales · 203,104 observaciones empresa-ano |
| **Caso de estudio** | Subconjunto de PYMES de Ibague (61 empresas via cruce CCI) |
| **Tarea ML** | Clasificacion multiclase de riesgo financiero (bajo/medio/alto) |
| **Modelo principal** | XGBoost (justificado en estado del arte v2 §3) |
| **Interpretabilidad** | TreeSHAP |
| **Convencion** | Entrenar con nacional (sin Ibague) → validar con Ibague |
| **Entregables finales** | (1) Informe en LaTeX, (2) Diapositivas Beamer, (3) Codigo + datos en repo |

---

## 1. Estado actual (no rehacer)

**Hecho** ✅:
- ETL nacional: `notebooks/01_etl_camara_comercio.ipynb`, `notebooks/02_etl_nacional_pymes.ipynb`
- 5 CSVs consolidados versionados via Git LFS en `data/processed/`
- Estado del arte v2 con 48 referencias: `docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex`
- Modulos Python:
  - `src/indicadores.py` — 18 ratios financieros + Z''-Score Altman para mercados emergentes
  - `src/etl_utils.py` — carga del consolidado + normalizacion de mojibake en 105 columnas
- Documentacion: `docs/DICCIONARIO_DATOS_ML.md`, `reports/REPORTE_ETL_NACIONAL.md`

**Lo que falta** (este plan): fases 1 a 11.

---

## 2. Convenciones tecnicas (de obligatorio cumplimiento)

### 2.1 Carga de datos

Siempre via `src.etl_utils.cargar_consolidado()`. Esa funcion aplica `normalizar_columnas()` que mapea los 105 nombres con mojibake (`ï¿½`) a ASCII puro. **Nunca** escribir mojibake en codigo.

```python
import sys; sys.path.insert(0, '..')
from src.etl_utils import cargar_consolidado
df = cargar_consolidado()  # 203,104 filas x 230 columnas (post-normalizacion)
```

### 2.2 Reproducibilidad

- `random_state=42` en todos los splits, SMOTE, modelos.
- Cada notebook empieza con celda de imports + `np.random.seed(42)`.
- Versionar los CSV de salida en `data/processed/` (estan en LFS via `*.csv` en `.gitattributes`).
- Modelos serializados en `models/` (gitignored — se regeneran). Usar `joblib.dump()`.

### 2.3 Convencion nacional vs Ibague

- **Conjunto Ibague** = NITs presentes en la Camara de Comercio que matchean el SIREM (61 empresas). Se identifica una sola vez en Fase 4 y se reserva como holdout.
- **Conjunto Nacional** = todas las demas PYMES (38,245 − 61 = 38,184). Sobre estas se entrena/valida/testea con split temporal.
- **Prohibido**: que las 61 empresas de Ibague aparezcan en el split de entrenamiento/validacion/test del modelo nacional. Verificar en cada fase con un assert.

### 2.4 Split temporal

- Train: 2016–2021
- Validation: 2022
- Test: 2023–2024
- **Nunca** usar K-Fold aleatorio sobre el panel completo (introduce leakage por empresa cruzando años).

### 2.5 Salidas estandarizadas

| Carpeta | Que va | Versionado |
|---|---|---|
| `data/processed/*.csv` | datasets derivados (indicadores, features, etiquetas) | Si (LFS) |
| `data/ibague/*.csv` | holdout de Ibague | Si (LFS) |
| `reports/figures/*.png` | figuras del informe | Si |
| `reports/tables/*.tex` | tablas LaTeX para el informe | Si |
| `reports/metrics/*.json` | metricas de modelos | Si |
| `models/*.joblib` | modelos serializados | No (gitignored) |
| `notebooks/N_*.ipynb` | notebooks numerados | Si |

### 2.6 Estilo de codigo

- PEP 8 (lineas ≤100 chars).
- Docstrings en español, codigo (variables, funciones) en español sin tildes (consistente con `src/`).
- **Sin** `Co-Authored-By: Claude` en commits.
- Commits atomicos por fase: `feat: fase N - descripcion`.

---

## 3. Objetivos

### 3.1 Objetivo general

Disenar e implementar un modelo de Machine Learning supervisado que clasifique el nivel de riesgo financiero (bajo, medio, alto) de PYMES colombianas a partir de sus estados financieros NIIF, y validar su comportamiento sobre las PYMES de Ibague como caso de estudio.

### 3.2 Objetivos especificos

1. Calcular un conjunto reproducible de 18 indicadores financieros sobre el universo SIREM 2016–2024.
2. Construir una etiqueta de riesgo trianguladade (Z''-Score Altman + reglas heuristicas) que sirva como ground truth supervisado.
3. Entrenar y comparar tres modelos: regresion logistica (baseline), random forest y XGBoost.
4. Evaluar el desempeño con metricas robustas al desbalance (F1-macro, AUC-PR) y validacion temporal estricta.
5. Interpretar las predicciones del mejor modelo via TreeSHAP, identificando los factores de decision globales y locales.
6. Validar el modelo entrenado nacionalmente sobre el subconjunto de PYMES de Ibague y discutir sus limitaciones.

---

## 4. Fases del trabajo

> **Ejecucion**: las fases 1–8 son tecnicas (notebooks). Las fases 9–11 son de redaccion. El orden dentro de cada bloque es estricto por dependencias de archivo.

---

### FASE 1 — Calculo de indicadores y analisis exploratorio

**Notebook**: `notebooks/03_indicadores_financieros.ipynb`
**Effort estimado**: 4–6 horas
**Depende de**: ETL completo (ya hecho)

#### 1.1 Inputs

- `data/processed/colombia_consolidado_pymes.csv` (203,104 × 230)
- `src/indicadores.py`, `src/etl_utils.py`

#### 1.2 Tareas

1. Cargar el consolidado con `cargar_consolidado()`.
2. Aplicar `calcular_todos(df)` → DataFrame con NIT_LIMPIO, ANIO + 18 indicadores + Z-Score + 2 zonas Altman.
3. **EDA por indicador**:
   - Estadisticos descriptivos (mean, mediana, p1, p99, std, % nulos).
   - Histogramas (escala log para los que tienen distribucion sesgada: ROA, ROE, rotacion_*, dias_*).
   - Boxplots por año fiscal (deteccion de efectos macro: COVID 2020-2021).
4. **Analisis de completitud**: tabla con % de no-nulos por indicador, ordenada de menor a mayor.
5. **Deteccion de outliers**: aplicar winsorizacion a percentil 1-99 sobre los indicadores ratio (no sobre `capital_trabajo` que esta en pesos).
6. **Correlaciones**: matriz de correlacion (Spearman, no Pearson — datos no normales) entre los 18 indicadores. Heatmap.
7. **Distribucion del Z''-Score**: histograma + 3 lineas verticales en {1.1, 2.6, p33, p66} para ver dispersion entre umbrales originales y empiricos.

#### 1.3 Outputs

- `data/processed/colombia_indicadores_pymes.csv` — 203K × ~25 cols.
- `reports/figures/01_distribuciones_indicadores.png`
- `reports/figures/02_correlacion_indicadores.png`
- `reports/figures/03_distribucion_zscore.png`
- `reports/figures/04_completitud_por_indicador.png`
- `reports/tables/completitud_indicadores.tex`

#### 1.4 Criterios de aceptacion

- [ ] El CSV resultante tiene exactamente 203,104 filas.
- [ ] Los 18 indicadores estan presentes; ninguno tiene 100% de nulos (si pasa, hay un bug en columna source).
- [ ] El histograma del Z-Score muestra una distribucion unimodal aproximadamente normal/log-normal con cola.
- [ ] Las 5 figuras se generan sin errores y se ven legibles a 300 DPI.

#### 1.5 Pitfalls conocidos

- **No** olvidar `cargar_consolidado(normalizar=True)` — si normalizar=False las columnas tienen mojibake y todos los indicadores van a fallar silenciosamente con KeyError o NaN.
- Los ratios pueden producir `inf` cuando se interpreta mal una division. Verificar que `_safe_div` en `indicadores.py` esta retornando NaN (no inf).
- Hay 14.6% de filas (29,701) sin metadatos. Para el EDA inicial no afecta, pero si en un grafico se filtra por sector va a haber un sesgo.

---

### FASE 2 — Etiquetado de riesgo (ground truth supervisado)

**Notebook**: `notebooks/04_etiquetado_riesgo.ipynb`
**Effort estimado**: 6–8 horas
**Depende de**: Fase 1

#### 2.1 Justificacion

El SIREM no incluye un campo "empresa quebrada Si/No" para todas las PYMES. La etiqueta debe construirse. Siguiendo el estado del arte v2 §2.3, usamos triangulacion:

1. **Z''-Score Altman para mercados emergentes** (heuristica financiera).
2. **Reglas heuristicas** sobre indicadores observables (margen neto < 0 sostenido, patrimonio negativo, cobertura intereses < 1, etc.).
3. **Cuartiles empiricos del dataset** (etiqueta robusta al sesgo de calibracion EE.UU. → Colombia).

La etiqueta final es la **interseccion** de las tres senales (acuerdo).

#### 2.2 Inputs

- `data/processed/colombia_indicadores_pymes.csv`

#### 2.3 Tareas

1. **Etiqueta A — Z''-Score por umbrales originales**: usar `clasificar_zona_altman()` (ya implementado).
2. **Etiqueta B — Z''-Score por terciles empiricos**: usar `clasificar_por_cuartiles()` (ya implementado).
3. **Etiqueta C — Reglas heuristicas combinadas**:

   ```
   def etiqueta_heuristica(row):
       senales_alto = 0
       senales_bajo = 0
       if row['margen_neto'] < 0: senales_alto += 1
       if row['razon_corriente'] < 1.0: senales_alto += 1
       if row['cobertura_intereses'] < 1.0 and pd.notna(row['cobertura_intereses']): senales_alto += 1
       if row['razon_deuda'] > 0.7: senales_alto += 1
       if row['capital_trabajo'] < 0: senales_alto += 1

       if row['margen_neto'] > 0.05: senales_bajo += 1
       if row['razon_corriente'] > 1.5: senales_bajo += 1
       if row['cobertura_intereses'] > 3.0: senales_bajo += 1
       if row['razon_deuda'] < 0.5: senales_bajo += 1
       if row['roa'] > 0.05: senales_bajo += 1

       if senales_alto >= 3: return 'riesgo_alto'
       if senales_bajo >= 3: return 'riesgo_bajo'
       return 'riesgo_medio'
   ```

4. **Tabla de concordancia**: cross-tab de A × B × C. Calcular acuerdo pairwise (Cohen's kappa).
5. **Etiqueta final**:
   - Si **B y C** coinciden: usar esa.
   - Si discrepan: marcar `riesgo_medio` (zona gris, conservador).
6. **Distribucion final** por año: tabla y grafico de barras apiladas.
7. **Validacion sectorial**: ¿hay sectores sobre-representados en `riesgo_alto`? Cruzar con CIIU.

#### 2.4 Outputs

- `data/processed/colombia_etiquetas_riesgo.csv` — NIT_LIMPIO, ANIO, etiqueta_A, etiqueta_B, etiqueta_C, etiqueta_final.
- `reports/figures/05_distribucion_etiquetas_por_ano.png`
- `reports/figures/06_etiquetas_por_sector.png`
- `reports/tables/concordancia_etiquetadores.tex`
- `reports/metrics/kappa_etiquetadores.json`

#### 2.5 Criterios de aceptacion

- [ ] Distribucion final tiene las 3 clases con minimo 5% cada una (si no, ajustar reglas heuristicas).
- [ ] Cohen's kappa entre B y C ≥ 0.4 (acuerdo moderado o mejor; si es muy bajo, las heuristicas son inconsistentes).
- [ ] El año 2020 muestra deterioro relativo (mayor proporcion de `riesgo_alto` o `riesgo_medio`) por COVID.

#### 2.6 Pitfalls

- `cobertura_intereses` es NaN cuando no hay costos financieros — interpretar como "no aplica", no como riesgo alto.
- No usar `etiqueta_A` (umbrales Altman originales) como etiqueta final — esta calibrada para EE.UU. y va a sobre-etiquetar `riesgo_alto` para Colombia.

---

### FASE 3 — Feature engineering temporal

**Notebook**: `notebooks/05_feature_engineering.ipynb`
**Effort estimado**: 3–5 horas
**Depende de**: Fase 1

#### 3.1 Tareas

1. **Variaciones interanuales** de los 18 indicadores: `(indicador_t - indicador_{t-1}) / |indicador_{t-1}|`. Manejar el primer año por empresa (NaN).
2. **Tasas de crecimiento** sobre 2 y 3 anos para los indicadores clave (ROA, margen_neto, razon_deuda).
3. **Log-tamano**: `log(1 + total_activos)`, `log(1 + ingresos)`.
4. **Estructura de balance**: `activos_corrientes / total_activos`, `pasivos_corrientes / total_pasivos`.
5. **Dummies categoricas** desde metadatos: sector CIIU (top 10 + "otros"), tipo organizacion (S.A.S., Ltda., S.A., otra), departamento (top 5 + "otros").
6. **Año fiscal** como variable numerica para capturar efectos macro (con dummy adicional para 2020 = COVID).

Total esperado: ~50–70 features tras encoding.

#### 3.2 Outputs

- `data/processed/colombia_features_ml.csv` — NIT_LIMPIO, ANIO, etiqueta_final, ~50–70 features.

#### 3.3 Criterios de aceptacion

- [ ] No hay columnas con 100% nulos.
- [ ] No hay duplicados (NIT_LIMPIO, ANIO).
- [ ] Las dummies suman 1 por fila dentro de cada grupo categorico (sanity check).

---

### FASE 4 — Particion train / val / test + holdout Ibague

**Notebook**: `notebooks/06_particion_datos.ipynb`
**Effort estimado**: 2 horas
**Depende de**: Fase 3 + cruce CCI ya hecho en notebook 01

#### 4.1 Tareas

1. Cargar lista de NITs de Ibague (extraer del notebook 01 cruce CCI ↔ SIREM).
2. **Holdout Ibague**: filtrar `colombia_features_ml.csv` por esos 61 NITs → `data/ibague/ibague_holdout.csv`.
3. **Set nacional**: el resto (38,245 − 61 = 38,184 empresas).
4. Sobre el set nacional, aplicar split temporal:
   - `train.csv`: ANIO ∈ [2016, 2021]
   - `val.csv`: ANIO == 2022
   - `test.csv`: ANIO ∈ [2023, 2024]
5. **Asserts criticos**:
   ```python
   assert len(set(ibague_nits) & set(train_nits)) == 0
   assert len(set(ibague_nits) & set(val_nits)) == 0
   assert len(set(ibague_nits) & set(test_nits)) == 0
   ```

#### 4.2 Outputs

- `data/processed/nacional_train.csv`
- `data/processed/nacional_val.csv`
- `data/processed/nacional_test.csv`
- `data/ibague/ibague_holdout.csv`
- `reports/tables/distribucion_clases_por_split.tex`

#### 4.3 Criterios de aceptacion

- [ ] Los 4 asserts pasan (no leakage Ibague).
- [ ] Distribucion de clases similar (±5%) entre train/val/test.
- [ ] Tamanos coherentes: train > val < test, todos > 1000 obs.

---

### FASE 5 — Modelado

**Notebook**: `notebooks/07_modelado.ipynb`
**Effort estimado**: 8–12 horas
**Depende de**: Fase 4

#### 5.1 Tareas

Entrenar **3 modelos** sobre `nacional_train.csv`, hiperparametrizar sobre `nacional_val.csv`, evaluar sobre `nacional_test.csv`.

##### Modelo 1: Regresion Logistica (baseline)

- `sklearn.linear_model.LogisticRegression(multi_class='multinomial', class_weight='balanced')`
- Estandarizar features (StandardScaler).
- Imputar NaN con la mediana (SimpleImputer).
- Sin SMOTE (logistic ya tiene `class_weight='balanced'`).
- Hiperparametros a buscar: `C ∈ {0.01, 0.1, 1, 10}` con grid search sobre validation.

##### Modelo 2: Random Forest

- `sklearn.ensemble.RandomForestClassifier(class_weight='balanced_subsample', random_state=42)`
- No requiere escalado.
- Imputar NaN con mediana.
- Hiperparametros: `n_estimators ∈ {100, 300, 500}`, `max_depth ∈ {None, 10, 20}`, `min_samples_split ∈ {2, 10}`.

##### Modelo 3: XGBoost (modelo principal)

- `xgboost.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, eval_metric='mlogloss')`
- Manejar desbalance via `sample_weight` (inversamente proporcional a frecuencia de clase) **dentro de cada fold de cross-validation**, no global.
- Comparar contra: SMOTE aplicado **solo dentro del pliegue de entrenamiento** (usar `imblearn.pipeline.Pipeline` con `SMOTE` antes del classifier; NUNCA aplicar SMOTE a todo el train antes de splitear).
- Hiperparametros: `n_estimators ∈ {200, 500, 1000}`, `max_depth ∈ {4, 6, 8}`, `learning_rate ∈ {0.05, 0.1}`, `subsample ∈ {0.8, 1.0}`, `colsample_bytree ∈ {0.8, 1.0}`.

##### Validacion

- Para hiperparametros: GridSearchCV sobre **validation set** (no CV aleatoria sobre train; el split temporal manda).
- Reportar el mejor modelo de cada familia con sus hiperparametros.

#### 5.2 Outputs

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/xgboost_smote.joblib`  (variante con SMOTE)
- `reports/metrics/best_hyperparams.json`
- `reports/tables/comparacion_modelos_validation.tex`

#### 5.3 Criterios de aceptacion

- [ ] Los 4 modelos serializados se cargan correctamente con `joblib.load`.
- [ ] El mejor XGBoost (con o sin SMOTE) supera al baseline logistico en F1-macro sobre `val.csv` por al menos +5 puntos porcentuales.
- [ ] Ningun modelo muestra signos obvios de overfitting (gap train F1 vs val F1 < 10pp).

#### 5.4 Pitfalls criticos

- **SMOTE leakage**: aplicar SMOTE despues del split, dentro del fold. Nunca a todo el train.
- **Imputacion leakage**: el `SimpleImputer` debe fitear con `train.csv` solamente.
- **Escalado leakage**: `StandardScaler.fit` solo con train.

---

### FASE 6 — Evaluacion + interpretabilidad TreeSHAP

**Notebook**: `notebooks/08_evaluacion_shap.ipynb`
**Effort estimado**: 6–8 horas
**Depende de**: Fase 5

#### 6.1 Tareas

##### Evaluacion sobre test (nacional)

Para cada modelo, sobre `nacional_test.csv`:
1. **Metricas globales**: accuracy, F1-macro, F1-ponderado, AUC-ROC OvR (one-vs-rest), AUC-PR macro.
2. **Matriz de confusion** normalizada por fila (recall por clase).
3. **Curvas ROC y PR** por clase.
4. **Reporte por clase**: precision, recall, F1, support.

##### Comparativa de los 3 modelos

Tabla resumen + grafico de barras agrupadas (precision/recall/F1 por modelo).

##### Interpretabilidad (solo el mejor modelo)

1. **TreeSHAP global**:
   - `shap.TreeExplainer(modelo)` sobre una muestra representativa (~5,000 obs de test).
   - **Beeswarm plot** con las 20 features mas influyentes.
   - **Bar plot** de importancia agregada.
2. **TreeSHAP local**:
   - 3 ejemplos por clase (9 totales): waterfall plot mostrando contribucion de cada feature a esa prediccion.
   - Elegir ejemplos representativos (cerca del centroide de cada clase) y un caso ambiguo (probabilidad cercana a 0.5).

#### 6.2 Outputs

- `reports/metrics/test_metrics.json` — metricas globales y por clase para los 3 modelos.
- `reports/figures/07_matriz_confusion_xgboost.png`
- `reports/figures/08_curvas_roc_pr.png`
- `reports/figures/09_comparacion_modelos.png`
- `reports/figures/10_shap_beeswarm.png`
- `reports/figures/11_shap_importance_bar.png`
- `reports/figures/12_shap_waterfall_ejemplos.png` (multipanel 3×3)
- `reports/tables/metricas_test_por_modelo.tex`

#### 6.3 Criterios de aceptacion

- [ ] El mejor modelo logra F1-macro ≥ 0.70 sobre el test (esperable dado el universo grande).
- [ ] Los SHAP global y los hallazgos del estado del arte v2 §3.4 (cobertura de intereses, razon deuda, margen neto como features clave) son consistentes — si no, discutir la divergencia.
- [ ] Los 9 waterfall plots se generan correctamente.

---

### FASE 7 — Validacion en Ibague (caso de estudio)

**Notebook**: `notebooks/09_validacion_ibague.ipynb`
**Effort estimado**: 4–6 horas
**Depende de**: Fase 6

#### 7.1 Tareas

1. Cargar `data/ibague/ibague_holdout.csv` y el mejor modelo XGBoost.
2. Aplicar el mismo preprocesamiento (imputacion + escalado fit-eados con `nacional_train`).
3. Predecir clases + probabilidades para las 61 PYMES × 9 anos.
4. **Comparacion contra ground truth heuristico** (etiqueta_final de Fase 2):
   - Matriz de confusion sobre Ibague.
   - F1-macro y por clase.
5. **Analisis de empresas individuales**:
   - 5 PYMES bien clasificadas (probabilidad > 0.8): perfil financiero + razon de la prediccion (SHAP local).
   - 5 PYMES mal clasificadas: indagar por que.
6. **Distribucion de riesgo en Ibague vs nacional**: comparar la proporcion de empresas en cada clase.
7. **Evolucion temporal**: para 3-5 PYMES con datos completos 2016–2024, graficar la trayectoria del riesgo predicho.

#### 7.2 Outputs

- `data/ibague/predicciones_ibague.csv` — NIT_LIMPIO, ANIO, prob_bajo, prob_medio, prob_alto, prediccion, etiqueta_real.
- `reports/figures/13_matriz_confusion_ibague.png`
- `reports/figures/14_perfil_5_pymes_bien_clasificadas.png`
- `reports/figures/15_perfil_5_pymes_mal_clasificadas.png`
- `reports/figures/16_evolucion_riesgo_ibague.png`
- `reports/tables/metricas_ibague.tex`

#### 7.3 Criterios de aceptacion

- [ ] El F1-macro en Ibague no se degrada mas de 10pp respecto al test nacional (si pasa, hay diferencia distribucional sectorial relevante para discutir).
- [ ] Cada empresa mal clasificada tiene una explicacion plausible escrita en el notebook (datos faltantes, sector atipico, año de crisis, etc.).

---

### FASE 8 — Discusion: comparacion contra metodos clasicos y revision manual

**Notebook**: `notebooks/10_discusion_comparativa.ipynb`
**Effort estimado**: 6–8 horas
**Depende de**: Fases 6 y 7

> Esta fase implementa la seccion "Discusion" que pidio el profesor: comparar contra otros metodos, resultados manuales y golden standard.

#### 8.1 Tareas

##### 8.1.1 vs Metodo clasico (Z''-Score Altman puro)

- Aplicar `clasificar_zona_altman()` (umbrales originales 1.1/2.6) y `clasificar_por_cuartiles()` directamente sobre el test nacional.
- Tratar cada uno como un "modelo" y calcular las mismas metricas que XGBoost (F1, AUC-ROC).
- Tabla comparativa: **XGBoost vs Altman umbrales originales vs Altman terciles vs Logistic vs RF**.

##### 8.1.2 vs Resultados manuales (golden standard)

- Seleccionar **30 PYMES de Ibague** estratificadamente (10 por clase predicha).
- **Etiquetado manual experto**: para cada empresa, leer sus indicadores y asignar manualmente una etiqueta. Justificar la decision en 2-3 oraciones por empresa.
- Calcular acuerdo entre etiqueta manual y prediccion XGBoost (Cohen's kappa, accuracy).
- Identificar patrones donde el modelo y el experto humano discrepan sistematicamente.

> **Nota operativa**: dado que los autores son los etiquetadores, documentar criterios explicitos antes del etiquetado para reducir sesgo. Idealmente etiquetar de forma ciega (sin ver la prediccion del modelo).

##### 8.1.3 vs Literatura

- Tabla comparativa: incluir la del estado del arte v2 §3 con XGBoost (Boumhidi 2025: AUC 0.93, Mahesh 2025: 92.7%, Yufenyuy 2024: 93.4%, Dasilas 2024 SLR: 89% promedio) **+ nuestra columna**.
- Discutir: ¿esta nuestro F1/AUC en el rango de la literatura? ¿que diferencias contextuales podrian explicar gaps?

##### 8.1.4 Analisis de robustez

- **Estabilidad temporal**: F1 sobre test 2023 vs F1 sobre test 2024. ¿Hay degradacion (signo de drift)?
- **Estabilidad sectorial**: F1 por sector CIIU (top 5 sectores). ¿Hay sectores donde el modelo falla sistematicamente?
- **Estabilidad de SHAP**: ejecutar 5 modelos XGBoost identicos con diferentes random seeds, comparar el ranking de top-10 features. Aplicar la metodologia de Lin (2024) citada en el SoTA v2 §6.

#### 8.2 Outputs

- `data/ibague/etiquetado_manual.csv` — 30 empresas con justificacion manual.
- `reports/figures/17_comparacion_xgboost_vs_clasicos.png`
- `reports/figures/18_estabilidad_sectorial.png`
- `reports/figures/19_estabilidad_shap_seeds.png`
- `reports/tables/comparacion_metodos.tex`
- `reports/tables/comparacion_literatura.tex`
- `reports/tables/etiquetado_manual_resumen.tex`

#### 8.3 Criterios de aceptacion

- [ ] La tabla `comparacion_metodos.tex` muestra que XGBoost supera a Altman puro por al menos 10pp en F1-macro.
- [ ] El acuerdo XGBoost vs etiquetado manual (kappa) es ≥ 0.5 (acuerdo moderado).
- [ ] La estabilidad SHAP de los top-3 features es ≥ 90% (estables; hallazgo consistente con Lin 2024).

---

### FASE 9 — Revision del estado del arte (ajuste post-resultados)

**Effort estimado**: 4–6 horas
**Depende de**: Fase 8

> El profesor pidio "modificarlo si es necesario". Despues de obtener resultados es probable que algunas afirmaciones del estado del arte v2 deban ajustarse o complementarse.

#### 9.1 Tareas

1. Releer `docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex` con los resultados en mano.
2. Revisar especialmente:
   - **§2.3 (Z-Score como heuristica de etiquetado)**: ¿se cumplieron las 3 mitigaciones? Si en Fase 8.1.1 XGBoost demostro superioridad sobre Altman puro, agregar este resultado empirico al final de la seccion como evidencia que valida el approach.
   - **§3.2 (XGBoost en mercados emergentes)**: agregar nuestra observacion empirica al cierre comparativo.
   - **§4 (Sistemas web)**: si el alcance del trabajo final IA excluye el sistema web, **acortar** esta seccion o moverla a "trabajo futuro".
   - **§7.2 (Validacion temporal)**: agregar nota sobre la implementacion concreta del split 2016-21/22/23-24.
3. Actualizar la **Tabla 2** (`tab:ml`) con la fila "Este trabajo" reemplazando `Por det.` con la metrica real.

#### 9.2 Outputs

- `docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex` actualizado.
- Compilar para verificar que no hay errores de bibliografia.

#### 9.3 Criterios de aceptacion

- [ ] El .tex compila sin warnings de citas faltantes.
- [ ] La tabla 2 tiene metricas reales en la fila final.
- [ ] La seccion 4 (sistemas web) esta acortada o relocada si no es relevante.

---

### FASE 10 — Redaccion del informe final (LaTeX)

**Carpeta**: `docs/informe_final/`
**Effort estimado**: 12–16 horas
**Depende de**: Fases 1–9 completas

#### 10.1 Estructura solicitada por el profesor

```
1. Titulo
2. Resumen (~200 palabras) + Abstract en ingles (~150 palabras)
3. Palabras clave (5-7) + Keywords
4. Introduccion
   4.1 Ubicacion del problema
   4.2 Referencias al estado del arte (resumir el v2 en 1-2 pp + citar)
   4.3 Justificacion
   4.4 Objetivo general y especificos
5. Metodologia
6. Materiales
7. Desarrollo
8. Resultados (figuras + tablas)
9. Discusion
   9.1 Comparacion contra otros metodos (XGBoost vs LR vs RF vs Altman)
   9.2 Resultados manuales / golden standard
   9.3 Comparacion con literatura
10. Conclusiones (mapeo a objetivos)
11. Referencias
```

#### 10.2 Mapeo seccion → fuente

| Seccion del informe | De donde sale el contenido |
|---|---|
| Titulo | "Prediccion de Riesgo Financiero en PYMES Colombianas mediante Machine Learning con Datos del SIREM (2016–2024)" |
| Resumen | Sintesis de objetivos + metodologia + resultado principal (F1, AUC) + conclusion en 200 palabras |
| Palabras clave | Machine Learning, riesgo financiero, PYMES, XGBoost, SHAP, NIIF, SIREM |
| 4.1 Ubicacion del problema | Estado del arte v2 §1 (riesgo en PYMES) + datos macro de Colombia (Cardona-Zuleta 2025, Higuera 2021) |
| 4.2 Estado del arte | **Resumen condensado** del v2 (1-2 pp). Citar las 7 secciones. Tabla comparativa de literatura del v2 §3. |
| 4.3 Justificacion | Vacios identificados en SoTA v2 §8 (escala datos + LATAM + integracion + interpretabilidad) |
| 4.4 Objetivos | Seccion 3 de este plan |
| 5. Metodologia | Pipeline (Fases 1-8) + decisiones metodologicas (split temporal, SMOTE, TreeSHAP) |
| 6. Materiales | SIREM (4 datasets crudos), CCI Ibague, herramientas (Python 3.x, sklearn, xgboost, shap) |
| 7. Desarrollo | Narrativa de las Fases 1-7 con detalles de implementacion clave |
| 8. Resultados | Figuras 7-12 (evaluacion + SHAP) + tabla `metricas_test_por_modelo` + tabla `metricas_ibague` |
| 9.1 Discusion vs metodos | Tabla `comparacion_metodos` + figura 17 |
| 9.2 Discusion vs golden standard | Tabla `etiquetado_manual_resumen` + figura 18 (estabilidad sectorial) |
| 9.3 Discusion vs literatura | Tabla `comparacion_literatura` |
| 10. Conclusiones | Recapitulacion: cada objetivo (3.2) → resultado obtenido + limitacion |
| 11. Referencias | `references_v2.bib` (las 48 ya curadas) |

#### 10.3 Tareas

1. Crear `docs/informe_final/informe_final.tex` con preambulo similar al `estado_del_arte_v2.tex`.
2. Crear `docs/informe_final/informe_final.bib` (copia + actualizaciones de `references_v2.bib`).
3. Escribir cada seccion siguiendo el mapeo 10.2.
4. Incluir figuras desde `reports/figures/` y tablas desde `reports/tables/`.
5. Compilar con `tectonic docs/informe_final/informe_final.tex`.
6. Iterar hasta que compile limpio (sin warnings de cita o referencia faltante).

#### 10.4 Outputs

- `docs/informe_final/informe_final.tex` (~25-35 paginas)
- `docs/informe_final/informe_final.bib`
- `docs/informe_final/informe_final.pdf` (compilado)

#### 10.5 Criterios de aceptacion

- [ ] El PDF se genera sin errores.
- [ ] Tiene exactamente las 11 secciones del 10.1.
- [ ] Cada figura del informe esta referenciada en el texto con `\ref{fig:N}`.
- [ ] Cada tabla esta referenciada con `\ref{tab:N}`.
- [ ] Bibliografia con entre 30 y 50 referencias (subconjunto curado del .bib).
- [ ] La seccion de Conclusiones lista explicitamente los 6 objetivos especificos y como se cumplio cada uno.

---

### FASE 11 — Diapositivas (Beamer)

**Archivo**: `docs/diapositivas/diapositivas.tex`
**Effort estimado**: 6–10 horas
**Depende de**: Fase 10

#### 11.1 Estructura sugerida (15–20 slides para presentacion 15-20 min)

```
1. Titulo + autores + universidad + asignatura
2. Agenda
3. Contexto del problema (2 slides)
   - PYMES en Colombia: peso economico, mortalidad, sub-investigacion
   - Brecha: literatura escasa en LATAM, sin integracion ML
4. Estado del arte resumido (1 slide con tabla comparativa)
5. Objetivos (1 slide con general + 6 especificos)
6. Metodologia (3 slides)
   - Pipeline diagram (Fases 1-8 visual)
   - Datos: 38,245 PYMES, 203K obs, NIIF Pymes
   - Split temporal + Ibague holdout
7. Etiquetado de riesgo (1 slide)
   - Triangulacion Z'' + heuristica + cuartiles
8. Modelado (1 slide)
   - LR vs RF vs XGBoost
9. Resultados (3 slides)
   - Tabla comparativa de modelos sobre test
   - SHAP global beeswarm
   - Validacion en Ibague
10. Discusion (2 slides)
    - vs metodos clasicos
    - vs literatura
11. Conclusiones (1 slide)
12. Limitaciones y trabajo futuro (1 slide)
13. Preguntas (1 slide cierre)
```

#### 11.2 Tareas

1. Crear `docs/diapositivas/diapositivas.tex` con `\documentclass{beamer}`.
2. Tema: `Madrid` o `Berlin` con colores de Universidad de Ibague (azul institucional).
3. Reutilizar figuras de `reports/figures/`.
4. Compilar con `tectonic`.

#### 11.3 Outputs

- `docs/diapositivas/diapositivas.tex`
- `docs/diapositivas/diapositivas.pdf`

#### 11.4 Criterios de aceptacion

- [ ] Entre 15 y 20 slides.
- [ ] Cada slide tiene maximo 7 lineas de texto (regla 7×7).
- [ ] El PDF compila sin errores.

---

## 5. Configuracion de LaTeX

### 5.1 Engine recomendado: tectonic

Tectonic es un engine LaTeX moderno, single-binary, que descarga paquetes on-demand. **No requiere sudo**.

```bash
# Instalacion (Linux x86_64)
mkdir -p ~/.local/bin && cd /tmp
curl -sL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C ~/.local/bin/ tectonic
chmod +x ~/.local/bin/tectonic
tectonic --version
```

### 5.2 Compilar

```bash
# Informe
tectonic docs/informe_final/informe_final.tex

# Diapositivas
tectonic docs/diapositivas/diapositivas.tex
```

### 5.3 Alternativa: TeXLive del sistema

```bash
sudo dnf install texlive-scheme-medium texlive-bibtex texlive-collection-langspanish
pdflatex docs/informe_final/informe_final.tex
biber docs/informe_final/informe_final
pdflatex docs/informe_final/informe_final.tex
pdflatex docs/informe_final/informe_final.tex
```

---

## 6. Cronograma sugerido y dependencias

```
                  Fase 1
                    │
                    ▼
         ┌──────► Fase 2 (etiquetado)
         │          │
         │          ▼
       Fase 3 ──► Fase 4 ──► Fase 5 ──► Fase 6 ──► Fase 7
       (FE)      (split)    (modelos) (eval+SHAP) (Ibague)
                                                    │
                                                    ▼
                                                  Fase 8 (discusion)
                                                    │
                                                    ▼
                                                  Fase 9 (ajuste SoTA)
                                                    │
                                                    ▼
                                                  Fase 10 (informe)
                                                    │
                                                    ▼
                                                  Fase 11 (slides)
```

### 6.1 Cronograma por sprint (suponiendo 4 sprints de 1 semana)

| Sprint | Fases | Entregable |
|---|---|---|
| 1 | 1, 2, 3 | Indicadores + etiquetado + features |
| 2 | 4, 5 | Particion + 3 modelos entrenados |
| 3 | 6, 7, 8 | Evaluacion + SHAP + Ibague + discusion |
| 4 | 9, 10, 11 | SoTA actualizado + informe + slides |

---

## 7. Quality gates entre fases

Antes de pasar a la siguiente fase, verificar:

| Gate | Verificacion |
|---|---|
| **G1** (post-Fase 1) | Los 18 indicadores estan calculados y la matriz de correlacion no muestra ninguna correlacion = 1.0 (excepto diagonal). |
| **G2** (post-Fase 2) | La etiqueta final tiene 3 clases con minimo 5% cada una. Kappa entre etiquetadores ≥ 0.4. |
| **G3** (post-Fase 4) | Los 4 asserts de no-leakage Ibague pasan. |
| **G4** (post-Fase 5) | XGBoost supera el baseline logistico en val por ≥ 5pp en F1-macro. |
| **G5** (post-Fase 6) | F1-macro de test ≥ 0.70 (esperable dado N grande). |
| **G6** (post-Fase 7) | F1 en Ibague no degrada > 10pp respecto a test nacional, o si lo hace, esta documentada la razon. |
| **G7** (post-Fase 8) | Tabla comparativa contiene al menos 5 metodos (LR, RF, XGBoost, Altman original, Altman terciles). |
| **G8** (post-Fase 10) | El PDF compila limpio, todas las figuras numeradas, todas las tablas numeradas, biblio sin warnings. |

---

## 8. Riesgos conocidos y mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|---|---|---|
| **Etiqueta poco distinguible** (modelo aprende heuristica) | Media | Reglas heuristicas amplias (Fase 2.3), 50+ features (Fase 3) que no son inputs directos del Z''-Score, validacion contra etiquetado manual (Fase 8). |
| **SMOTE leakage** | Alta sin atencion | Pipeline de imblearn con SMOTE antes del classifier; `Pipeline.fit(X_train)` solo, validar con `cross_val_score`. |
| **Imputacion leakage** | Alta sin atencion | `SimpleImputer.fit(X_train)` y luego `.transform(X_val/test)`. |
| **Drift temporal** | Media | Reportar F1 separado por año en test; si hay degradacion en 2024, discutir como limitacion. |
| **Cuota LFS GitHub** | Baja | 1 GB free, ya consumimos 384 MB. Evitar re-pushes innecesarios. |
| **No disponibilidad de TeXLive** | Media | Tectonic como alternativa portable. Documentado en §5.1. |
| **Encoding mojibake re-aparece** en outputs nuevos | Media | Aplicar `normalizar_columnas()` antes de cualquier `to_csv()`. |
| **Bibliografia inconsistente** | Media | Reusar `references_v2.bib` y agregar entradas con cuidado; compilar con `tectonic` que reporta refs faltantes. |

---

## 9. Referencias clave del estado del arte v2

Cuando se necesite justificar una decision metodologica, citar:

| Decision | Citas en SoTA v2 |
|---|---|
| Etiquetar con Z''-Score Altman emergentes | §2.3 (Altman 1968, Wu 2022) |
| Mitigaciones del riesgo de etiqueta heuristica | §2.3 (Grice 2003, Bouwmeester 2020, Qiu 2020) |
| XGBoost como modelo principal | §3.2 (Boumhidi 2025, Mahesh 2025, Sujatha 2025, Dasilas 2024) |
| TreeSHAP para interpretabilidad | §6.1, §6.3 (LundbergTreeSHAP 2020, Pathi 2025, Lin 2024, Deng 2024) |
| SMOTE solo dentro de pliegues | §3.4 (Sun 2020, Kristanti 2025) |
| Validacion temporal estricta | §7.3 (Bortolotti 2024, Mahesh 2025, Zhou 2022) |
| Sesgo de seleccion del SIREM | §5.4 (Cardona-Zuleta 2025, Zhang-Zhang 2026) |

---

## 10. Comandos utiles

### Ejecutar un notebook desde linea de comando

```bash
cd notebooks/
jupyter nbconvert --to notebook --execute 03_indicadores_financieros.ipynb \
    --ExecutePreprocessor.timeout=1800 \
    --output 03_indicadores_financieros.ipynb
```

### Sanity check de un CSV procesado

```python
from src.etl_utils import cargar_consolidado, auditar_mojibake
df = cargar_consolidado()
print(df.shape)
print('Mojibake restante:', auditar_mojibake(df))
```

### Compilar el informe LaTeX

```bash
tectonic docs/informe_final/informe_final.tex
# Output: docs/informe_final/informe_final.pdf
```

### Crear commit por fase

```bash
git add notebooks/03_indicadores_financieros.ipynb \
        data/processed/colombia_indicadores_pymes.csv \
        reports/figures/0[1-4]_*.png \
        reports/tables/completitud_indicadores.tex
git commit -m "feat: fase 1 - calculo de 18 indicadores + EDA"
git push
```

---

## 11. Checklist global

### Tecnico
- [ ] Fase 1 — Indicadores calculados + EDA completo
- [ ] Fase 2 — Etiqueta de riesgo trianguladade
- [ ] Fase 3 — Features con FE temporal
- [ ] Fase 4 — Particion sin leakage Ibague
- [ ] Fase 5 — 3 modelos entrenados (LR, RF, XGBoost)
- [ ] Fase 6 — Evaluacion + SHAP global y local
- [ ] Fase 7 — Validacion en Ibague
- [ ] Fase 8 — Comparacion contra metodos clasicos + golden standard

### Documental
- [ ] Fase 9 — Estado del arte ajustado con resultados
- [ ] Fase 10 — Informe en LaTeX compilado
- [ ] Fase 11 — Diapositivas Beamer compiladas

### Repo
- [ ] Todos los commits con mensajes claros, sin Co-Authored-By Claude
- [ ] README.md actualizado con estado final del proyecto
- [ ] PDFs finales (informe + slides) commiteados en `docs/informe_final/` y `docs/diapositivas/`

---

*Documento creado: Mayo 2026 · Plan de trabajo - Universidad de Ibague*
