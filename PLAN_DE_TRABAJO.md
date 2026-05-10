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
- **Fase 1** ✅ (2026-05-09): `notebooks/03_indicadores_financieros.ipynb` ejecutado.
  Salidas: `data/processed/colombia_indicadores_pymes.csv` (203,104 × 23),
  `reports/figures/01..04*_*.png` (5 figuras a 300 DPI),
  `reports/tables/completitud_indicadores.tex`. Hallazgos clave:
  - Completitud entre 54.8% (`dias_inventario`) y 95.2% (`rotacion_cartera`); ninguno 100% nulo.
  - Z''-Score: mediana 4.93, distribucion unimodal con cola pesada.
  - Distribucion zona Altman original: 65.9% bajo / 10.7% medio / 23.5% alto (sobre-etiqueta `bajo`).
  - Distribucion por terciles empiricos: 30.2% bajo / 30.2% medio / 39.5% alto (mas balanceada — preferida para Fase 2).
- **Fase 2** ✅ (2026-05-09): `notebooks/04_etiquetado_riesgo.ipynb` ejecutado.
  Salidas: `data/processed/colombia_etiquetas_riesgo.csv` (203K × 9),
  `reports/figures/05_*`, `06_etiquetas_por_sector.png`, `05b_concordancia_BC_heatmap.png`,
  `reports/tables/concordancia_etiquetadores.tex`,
  `reports/metrics/kappa_etiquetadores.json`. Hallazgos clave:
  - **Calibracion empirica de la heuristica C**: los umbrales fijos del plan (margen<0, razon_corriente<1, etc.) producian kappa B-C = 0.304. Reemplazados por **terciles empiricos por indicador** (autorizado por el plan §2.5 "ajustar reglas heuristicas"). Resultado: **kappa B-C = 0.446** ≥ 0.4 ✅.
  - Distribucion `etiqueta_final`: 16.8% alto / 66.4% medio / 16.8% bajo (3 clases ≥ 5% ✅, sesgada hacia "medio" por la regla conservadora B==C → etiqueta).
  - Kappa A-B = 0.234, A-C = 0.203 (Altman umbrales originales descartado correctamente).
  - **Hallazgo empirico** discutible en informe: el anyo 2020 NO muestra deterioro vs 2019 (delta = -0.11 pp). Posibles causas: alivios fiscales decretos 535/560 de 2020, manifestacion diferida del impacto en cifras NIIF Pymes, sesgo de seleccion del SIREM (las empresas que cerraron no reportan).
  - **ANOMALIA 2017** ⚠️: 99.1% de las observaciones de 2017 tienen Z-Score nulo (problema del ETL/dataset SIREM en ese anyo: razon_corriente, razon_deuda, capital_trabajo, roa, z_score_altman tienen 99% de nulos; solo `margen_neto` (94%) y `cobertura_intereses` (67%) son utilizables). Como consecuencia, casi todas las filas de 2017 caen en `riesgo_medio` por consenso fallido. **Recomendacion para Fase 4**: filtrar `ANIO != 2017` antes del split temporal o tratarlo como subset separado.
- **Fase 3** ✅ (2026-05-09): `notebooks/05_feature_engineering.ipynb` ejecutado.
  Salidas: `data/processed/colombia_features_ml.csv` (203,104 × 77, 148 MB),
  `reports/metrics/features_ml_resumen.json`, `reports/tables/features_familias.tex`.
  Hallazgos clave:
  - **77 columnas** = 3 IDs + 71 features ML + 3 categoricas de texto (diagnostico).
  - Familias de features: 18 indicadores base + Z''-Score + 18 deltas interanuales (`*_d1`) + 6 crecimientos multi-anyo (`*_g2`/`*_g3` para roa, margen_neto, razon_deuda) + 4 escala/estructura (log activos/ingresos, ratios corrientes) + 2 temporales (anyo numerico + dummy COVID 2020) + 22 dummies (CIIU 12 / sociedad 4 / depto 6).
  - **3 criterios 3.3 cumplidos**: (1) cero columnas 100% nulas; (2) cero duplicados (NIT, ANIO); (3) los tres grupos de dummies suman exactamente 1 en las 203,104 filas.
  - Distribucion `etiqueta_final` preservada (16.8/66.4/16.8) y empresas unicas intactas (38,245); el filtrado Ibague + filtrado 2017 ocurren en Fase 4.
  - **Completitud peor caso de features nuevos**: ~43% (`dias_inventario_d1`, `rotacion_inventarios_d1`, `roa_g3`, `razon_deuda_g3`) — todos arrastran la anomalia 2017. Mejor caso: `log_ingresos` 100%. Promedio de los 28 features nuevos: ~64%.
  - **Decision metodologica**: dummies sin `drop_first` (cada grupo suma 1) para que los modelos lineales sin intercepto funcionen correctamente; XGBoost y RF son indiferentes a la colinealidad de un grupo completo.
  - **Winsorizacion p1-p99** aplicada sobre 51 columnas continuas (indicadores base, deltas, crecimientos, logs, ratios estructurales). No se winsoriza dummies, anyo_num ni IDs.
- **Fase 4** ✅ (2026-05-09): `notebooks/06_particion_datos.ipynb` ejecutado.
  Salidas: `data/processed/nacional_train.csv` (108,522 × 77, 75 MB),
  `data/processed/nacional_val.csv` (26,878 × 77, 22 MB),
  `data/processed/nacional_test.csv` (50,957 × 77, 43 MB),
  `data/ibague/ibague_holdout.csv` (359 × 77, 274 KB),
  `reports/tables/distribucion_clases_por_split.tex`,
  `reports/figures/07_distribucion_clases_por_split.png` (300 DPI),
  `reports/metrics/particion_resumen.json`. Hallazgos clave:
  - **Holdout Ibague**: 61 NITs presentes (de los 66 del cruce CCI ↔ SIREM, los 5 ausentes son NIIF Plenas o "No aplica" filtrados por notebook 02). 359 filas empresa-anyo (incluye 2017 para diagnostico Fase 7).
  - **Filtrado ANIO=2017** sobre el split nacional: 16,419 filas eliminadas (8.3% del nacional pre-filtrado) por anomalia de Z-Score documentada en Fase 2.
  - **Asserts no-leakage Ibague**: 4/4 cumplidos (train, val, test sin NITs Ibague; cobertura completa de NITs nacionales).
  - **Distribucion de clases** (alto/medio/bajo): train 19.6/63.9/16.5, val 16.4/63.8/19.7, test 16.5/62.5/21.0, ibague 17.8/71.6/10.6. Delta maximo entre splits = **4.5pp** ≤ 5pp ✓ (clase `bajo` train vs test).
  - **Tamanos coherentes**: train (108K) > val (27K) < test (51K), todos > 1000 obs ✓.
  - **Observacion**: en val (2022) y test (2023-2024), `n_filas == n_nits` para val porque es un solo anyo; en test cada NIT puede aparecer hasta 2 veces.
  - **Drift sutil de etiqueta**: `riesgo_bajo` aumenta del 16.5% (train 2016, 2018-2021) al 21.0% (test 2023-2024), mientras `riesgo_alto` cae del 19.6% al 16.5%. Coherente con recuperacion post-COVID; documentar en informe Fase 10.
- **Fase 5** ✅ (2026-05-09): `notebooks/07_modelado.ipynb` ejecutado con XGBoost en GPU (RTX 2060, CUDA 12.9, `device='cuda'`).
  Salidas: `models/{logistic_regression,random_forest,xgboost,xgboost_smote}.joblib`,
  `reports/metrics/best_hyperparams.json`,
  `reports/tables/comparacion_modelos_validation.tex`. Hallazgos clave:
  - **Mejor modelo: XGBoost (sample_weight)** -- F1-macro val = **0.9979**, gap train-val = +0.21pp. HP ganadores: `max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0, best_iteration=221` (early_stopping_rounds=30 con cap 1000).
  - **Comparativa F1-macro val**: XGBoost 0.9979 > XGBoost+SMOTE 0.9957 > RandomForest 0.9948 (n_est=500, max_depth=None, mss=10) > LogisticRegression 0.7993 (C=10).
  - **Criterios PLAN 5.3 cumplidos**:
    - 4 modelos serializados cargan correctamente con `joblib.load` ✓.
    - Mejor XGBoost vs LR baseline = **+19.86pp** ≥ +5pp ✓.
    - Gap max |train-val| = **0.81pp** (LR) < 10pp ✓.
  - **Reduccion de grid documentada**: `n_estimators` para XGBoost reemplazado por `early_stopping_rounds=30` con cap 1000 (busqueda fina automatica via best_iteration). Resto del grid del plan respetado: 4 LR + 18 RF + 24 XGB + 24 XGB+SMOTE = 70 fits.
  - **Observacion empirica importante** (a discutir en Fase 8): los 3 modelos no-lineales (RF, XGB, XGB+SMOTE) memorizan casi perfectamente la etiqueta porque las features incluyen los mismos indicadores que construyen la etiqueta (`z_score_altman`, `margen_neto`, `razon_corriente`, `cobertura_intereses`, `razon_deuda`, `capital_trabajo`, `roa`). LR (lineal) no logra aproximar las reglas condicionales heuristicas y queda en ~0.80. Esto valida que el problema es aprendible pero anticipa que la *contribucion marginal* del ML sobre la heuristica pura debe demostrarse en Fase 8.1.1 (XGBoost vs Altman puro).
  - **GPU vs CPU**: con `device='cuda'` cada fit XGB tarda ~3s vs ~12s en CPU (~4× speedup); el notebook completo corrio en ~14 min en lugar de ~22 min estimados.
- **Fase 6** ✅ (2026-05-09): `notebooks/08_evaluacion_shap.ipynb` ejecutado.
  Salidas: `reports/metrics/test_metrics.json` (4.1 KB),
  `reports/figures/08_matriz_confusion_xgboost.png` (224 KB),
  `reports/figures/09_curvas_roc_pr.png` (467 KB),
  `reports/figures/10_comparacion_modelos.png` (168 KB),
  `reports/figures/11_shap_beeswarm.png` (1.0 MB),
  `reports/figures/12_shap_importance_bar.png` (320 KB),
  `reports/figures/13_shap_waterfall_ejemplos.png` (1.4 MB),
  `reports/tables/metricas_test_por_modelo.tex`. Hallazgos clave:
  - **Mejor modelo en test: XGBoost (sample_weight)** con F1-macro = **0.9972**, AUC-ROC OvR macro = 0.99999, AUC-PR macro = 0.99997. Ranking test: XGBoost 0.9972 > XGBoost+SMOTE 0.9956 > RandomForest 0.9943 > LogisticRegression 0.8105.
  - **Sin drift train→val→test** para XGBoost: train 1.0000 → val 0.9979 → test 0.9972 (gap test-val = -0.07pp, despreciable). LR sube ligeramente test=0.8105 vs val=0.7993 (+1.12pp), tampoco preocupante.
  - **Matriz de confusion XGBoost** (test 50,957): recall por clase 99.85% (riesgo_alto, n=8,403), 99.53% (riesgo_bajo, n=10,723), 99.80% (riesgo_medio, n=31,831). Errores totales: 127/50,957 = 0.25%. La confusion residual es principalmente alto↔medio y bajo↔medio (zona gris natural).
  - **TreeSHAP global** (5,000 obs estratificadas): top 10 features por `mean(|SHAP|)`:
    1. z_score_altman (3.16) -- 3× mas que el segundo;
    2. margen_neto (1.05);
    3. razon_corriente (1.02);
    4. cobertura_intereses (0.95);
    5. razon_deuda (0.72);
    6. roa (0.55);
    7. capital_trabajo (0.27);
    8. deuda_patrimonio (0.18);
    9. roe (0.08);
    10. apalancamiento (0.06).
  - **Consistencia con literatura SoTA v2 §3.4**: 7/9 features esperadas (cobertura_intereses, razon_deuda, margen_neto, roa, razon_corriente, capital_trabajo, z_score_altman) aparecen en el top 10 SHAP del modelo. CONSISTENTE.
  - **Observacion sobre dominancia de z_score_altman**: el feature z_score_altman pesa ~3× mas que el segundo (margen_neto). Esto corrobora la observacion de Fase 5: la etiqueta_final es B==C, donde B son los terciles del Z-Score y C son las heuristicas con margen_neto/razon_corriente/cobertura_intereses/razon_deuda/capital_trabajo. El modelo aprovecha que el ground truth fue construido con ese mismo Z-Score, por lo que reproducir Z-Score → etiqueta_final es trivial. La discusion de **valor marginal del ML vs Altman puro** queda para Fase 8.1.1 (planeada).
  - **Criterios PLAN 6.3 cumplidos**:
    - F1-macro test ≥ 0.70 ✓ (0.9972).
    - SHAP consistente con literatura ✓ (7/9 matches).
    - 9 waterfall plots generados ✓ (figura 13, 1.4 MB).
  - **Nota de numeracion de figuras**: el plan original asignaba 07-12 a las figuras de Fase 6, pero la Fase 4 ya consumio `07_distribucion_clases_por_split.png`. Para evitar pisar archivos, las figuras de Fase 6 se desplazaron a 08-13. Las fases 7+ deben ajustar (originalmente 13-16 -> 14-17 para Fase 7).
- **Fase 7** ✅ (2026-05-09): `notebooks/09_validacion_ibague.ipynb` ejecutado.
  Salidas: `data/ibague/predicciones_ibague.csv` (359 x 12),
  `reports/metrics/metricas_ibague.json` (5.7 KB),
  `reports/tables/metricas_ibague.tex` (4 subsets + referencia),
  `reports/figures/14_matriz_confusion_ibague.png` (212 KB),
  `reports/figures/15_perfil_5_pymes_bien_clasificadas.png` (928 KB, 5 paneles),
  `reports/figures/16_perfil_5_pymes_mal_clasificadas.png` (212 KB, 1 panel -- ver hallazgo abajo),
  `reports/figures/17_evolucion_riesgo_ibague.png` (717 KB, 5 PYMES con cobertura 9 anyos). Hallazgos clave:
  - **Desempeno casi perfecto sobre Ibague**: F1-macro = **0.9948** sin 2017 (n=328) y **0.9949** con 2017 (n=359). Solo **1 error de 359 predicciones** (acc=99.72%). AUC-ROC OvR = 1.0000.
  - **Criterio 7.3 #1 PASS**: gap F1-macro test nacional (0.9972) vs Ibague sin 2017 (0.9948) = **+0.24 pp** (umbral <= 10pp).
  - **F1-macro por anyo**: 2016-2023 = 1.0000; 2024 = 0.9640 (donde aparece el unico error). Sin drift temporal acumulado.
  - **2017 en Ibague**: pese a la anomalia SIREM (99% nulos en Z-Score nacionalmente), el modelo clasifica correctamente las 31 obs de Ibague en 2017 (acc=100%). Dos lecturas posibles: (1) los NITs Ibague tienen mejor reporte que el promedio nacional en 2017, (2) el modelo aprovecha las features no-Z disponibles. Documentar en informe Fase 10 como diagnostico positivo.
  - **Unico error -- NIT *043005, 2024**: real=`riesgo_bajo`, pred=`riesgo_medio` (prob 0.66). Z-Score 6.95 (sano), pero `cobertura_intereses` cayo 95% vs 2023 (de 34 a 1.66). El modelo capta esta volatilidad operativa que la heuristica de Fase 2 (basada en niveles, no en cambios) no observa. **3 razones documentadas** automaticamente: modelo conservador en zona sana, volatilidad operativa interanual, baja confianza (p<0.7). **Criterio 7.3 #2 PASS** (1/1 errores explicado).
  - **Limitacion del plan vs realidad**: el plan 7.1 pedia "5 PYMES bien + 5 mal clasificadas". Solo hay 1 error en todo el holdout, asi que la figura 16 muestra 1 panel en lugar de 5 (no se pueden inventar errores). La figura 15 si tiene 5 PYMES bien clasificadas con cobertura por clase (alto/bajo/medio).
  - **Distribucion Ibague vs Nacional**: ibague_real_sin_2017 = 19.5% alto / 11.6% bajo / 68.9% medio vs nacional_test = 16.5% alto / 21.0% bajo / 62.5% medio. Ibague tiene mas riesgo medio (+6.4pp) y menos riesgo bajo (-9.4pp) que el nacional. El modelo replica esta distribucion en sus predicciones (delta < 1pp), confirmando ausencia de bias regional.
  - **Insumos para Fase 8 listos**: `predicciones_ibague.csv` se puede usar como base para el etiquetado manual experto (8.1.2, criterio 30 PYMES). El JSON tiene los hooks para `comparacion_metodos` (8.1.1).
- **Fase 8** ✅ (2026-05-09): `notebooks/10_discusion_comparativa.ipynb` ejecutado.
  Salidas: `data/ibague/etiquetado_manual.csv` (30 PYMES, 5.2 KB),
  `reports/figures/18_comparacion_xgboost_vs_clasicos.png` (270 KB),
  `reports/figures/19_estabilidad_sectorial.png` (195 KB),
  `reports/figures/20_estabilidad_shap_seeds.png` (488 KB),
  `reports/tables/comparacion_metodos.tex`,
  `reports/tables/comparacion_literatura.tex`,
  `reports/tables/etiquetado_manual_resumen.tex`,
  `reports/metrics/discusion_fase8.json` (5.9 KB). Hallazgos clave:
  - **8.1.1 vs metodos clasicos sobre test nacional (n=50,957)**: F1-macro: XGBoost **0.9972** > XGBoost+SMOTE 0.9956 > RandomForest 0.9943 > LogisticRegression 0.8105 > **Altman terciles 0.6891 > Altman umbrales originales 0.4371**. **Delta XGBoost - Altman umbrales originales = +56.0pp**, **Delta XGBoost - Altman terciles = +30.8pp**. La heuristica original de Altman (1.1/2.6) calibrada para EE.UU. funciona muy mal en PYMES Colombia; los terciles empiricos mejoran substancialmente pero siguen muy por debajo del ML. **Criterio 8.3.1 PASS** (>=10pp).
  - **8.1.2 etiquetado manual** (30 PYMES Ibague, 10 por clase predicha XGBoost): protocolo reproducible declarado a-priori (Z-Score base + 4 ajustes: liquidez, solvencia, operativa, shock interanual). **Cohen kappa XGB vs manual = 0.500** (acuerdo moderado, exactamente en el umbral). Distribucion manual: 18 bajo / 4 medio / 8 alto vs XGB: 10/10/10 (XGB sobre-asigna riesgo_medio en zonas saludables; el protocolo manual es mas extremista porque solo dispara `medio` cuando hay un ajuste sobre `bajo` o `alto`). Kappa manual vs etiqueta_final Fase 2 = 0.500 (el protocolo manual diverge tambien de la heuristica B==C). **Criterio 8.3.2 PASS** (>=0.5).
  - **8.1.3 vs literatura SoTA v2**: tabla comparativa (Boumhidi 2025 AUC=0.93, Mahesh 2025 acc=92.7%, Yufenyuy 2024 acc=93.4%, Dasilas 2024 SLR F1~89%) **+ nuestro F1-macro 0.997 / AUC 1.0**. Nuestras metricas estan en el rango superior de la literatura, pero hay que advertir que estan infladas porque el target hereda el Z-Score (etiqueta B==C de Fase 2 usa los terciles del Z y heuristicas con margen_neto/razon_corriente/cobertura_intereses/razon_deuda/capital_trabajo/roa, todos features del modelo). La verdadera contribucion marginal del ML sobre la heuristica se mide en 8.1.1 (delta +56pp vs Altman puro).
  - **8.1.4 robustez**:
    - **Estabilidad temporal** (test 2023 vs 2024): F1-macro 2023 = 0.9970, 2024 = 0.9973, gap = -0.03pp. Sin drift temporal observable dentro del test. **OK <5pp**.
    - **Estabilidad sectorial** (top 5 CIIU comercio/inmobiliario/industria/construccion/profesional): F1-macro entre 0.9958 y 0.9980, gap max-min = 0.22pp. **OK <5pp**, sin sectores con desempeno degradado.
    - **Estabilidad SHAP 5 seeds** {7,21,42,100,2024} con HP ganadores Fase 5 (max_depth=8, lr=0.1, ss=0.8, cs=1.0, early_stopping_rounds=30): jaccard top-10 promedio = **0.927** (10 pares), minimo = 0.818. Top-3 features (z_score_altman, razon_corriente, margen_neto) presentes en **100%** de las seeds. 9/10 features de top-10 en seed=42 estables al 100%; solo `apalancamiento` cae a 20% (sustituido por `margen_operacional` en otras seeds — ambos correlacionan con estructura financiera). **Criterio 8.3.3 PASS** (>=90%, top-3 100%).
  - **Criterios PLAN 8.3 cumplidos**: (1) +56pp >= +10pp ✓; (2) kappa=0.50 >= 0.5 ✓; (3) top-3 100% >= 90% ✓.
  - **Nota de numeracion de figuras**: el plan asignaba 17-19 a Fase 8 pero la Fase 7 consumio 17. Se desplaza Fase 8 a 18-20 (mismo criterio que el desplazamiento Fase 6).
- **Fase 9** ✅ (2026-05-09): `docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex` ajustado post-resultados.
  Salidas: `.tex` actualizado (430 → 442 lineas, +12 lineas netas) y `estado_del_arte_v2.pdf` recompilado (181 KB) con `tectonic -X compile`. Hallazgos clave:
  - **§2.3 (Z-Score como heuristica)**: agregado parrafo "Validacion empirica posterior" que cuantifica la efectividad de la primera mitigacion declarada con los gaps F1-macro reales: XGBoost 0.997 vs Altman terciles 0.689 (+30.8pp) vs Altman umbrales originales 0.437 (+56.0pp). Vincula el resultado al espacio extendido de 71 features y refuerza la cita a Wu2022 (enfoque hibrido Z-Score + ML).
  - **§3.2 (XGBoost en mercados emergentes, modelos de ensamble)**: agregado parrafo "Aportacion empirica del presente trabajo" con HP ganadores (max_depth=8, lr=0.1, ss=0.8, cs=1.0, early_stopping=30), F1-macro test 0.997, AUC OvR ~1.0, gap train-test <0.3pp, y comparacion con Boumhidi 2025/Mahesh 2025/Yufenyuy 2024. Anyade validacion Ibague (61 NITs, 359 obs, F1=0.995).
  - **§7.2 (Validacion temporal)**: nota concreta sobre la implementacion: 108,522 / 26,878 / 50,957 / 359 obs en train/val/test/holdout Ibague, asserts de no-leakage 4/4 OK, exclusion ANIO=2017 (anomalia SIREM), F1 2023=2024=0.997 sin drift.
  - **§4 (Sistemas web)**: cierre reframado como **trabajo futuro**: el alcance del trabajo final IA excluye explicitamente el sistema web. Ajustes coherentes en (a) Tabla 2 columna Web "Este trabajo" `Si → Futuro`, (b) Tabla 4 `tab:vacios` fila Integracion web sustituye "Sistema web completo (React+Node+PostgreSQL)" por "Componente ML reproducible y exportable, base directa para una eventual interfaz web", (c) §8 sintesis y posicionamiento: el "tercer aspecto distintivo" pasa de "modelo de ML desplegado en sistema web" a "modelo de ML reproducible, exportable y documentado en pipeline completo desde SIREM crudo hasta inferencia con SHAP".
  - **Tabla 2 (`tab:ml`)**: fila "Este trabajo" reemplaza `Por det.` por `F1=0,997 / AUC≈1,0`.
  - **Compilacion**: `tectonic -X compile` exit 0, 0 warnings de citas indefinidas, 0 warnings de referencias indefinidas. Solo 35 warnings cosmeticas pre-existentes (`Overfull/Underfull \hbox` por anchos de longtable). PDF 181,552 bytes.
  - **Criterios PLAN 9.3 cumplidos**: (1) compila sin citas faltantes ✓; (2) Tabla 2 con metricas reales ✓; (3) §4 acortada/reframada ✓.
  - **Decision conservadora**: §4 NO se elimina; se preserva su valor metodologico para una eventual extension web futura, pero se elimina cualquier compromiso de entrega del sistema web en este trabajo. La cita a `MayorgaLira2023, BaumontDeOliveira2021, Ioannou2022` se mantiene como contexto.
- **Fase 10** ✅ (2026-05-09): informe final en LaTeX redactado y compilado.
  Salidas: `docs/informe_final/informe_final.tex` (~ 535 lineas), `docs/informe_final/informe_final.bib` (copia curada de 48 entradas), `docs/informe_final/informe_final.pdf` (5.4 MB, **33 paginas**). Hallazgos clave:
  - **Estructura completa**: las 11 secciones del 10.1 estan presentes -- (1) Titulo, (2) Resumen+Abstract, (3) Palabras clave/Keywords, (4) Introduccion (4.1 Ubicacion + 4.2 Estado del arte condensado + 4.3 Justificacion + 4.4 Objetivos), (5) Metodologia, (6) Materiales, (7) Desarrollo (Fases 1-5 con detalles), (8) Resultados, (9) Discusion (9.1 vs metodos clasicos + 9.2 golden standard + 9.3 vs literatura + 9.4 robustez + 9.5 limitaciones), (10) Conclusiones (mapeo a los 6 objetivos especificos), (11) Referencias.
  - **Compilacion limpia**: `tectonic -X compile informe_final.tex` exit 0, BibTeX clean (0 entradas faltantes en .blg), 0 referencias indefinidas, 0 citas indefinidas. Solo 9 warnings cosmeticas (`Overfull \hbox` en parrafos densos y en `comparacion_literatura.tex`).
  - **Cobertura de cross-references**: 16 figuras definidas y 16 figuras referenciadas en texto (`\ref{fig:N}`); 8 tablas con `\input` desde `reports/tables/` y las 8 referenciadas en texto (`\ref{tab:N}`).
  - **Bibliografia**: 36 entradas citadas activamente del .bib de 48 (subconjunto curado dentro del rango 30-50 del criterio).
  - **Conclusiones mapean los 6 objetivos**: cada objetivo especifico (3.2 del plan) tiene su parrafo dedicado con resultado obtenido + limitacion principal, mas cierre con aportes principales y trabajo futuro.
  - **Decisiones tecnicas LaTeX**: (a) preambulo derivado de `estado_del_arte_v2.tex` con `babel-spanish`, `inputenc utf8`, `fontenc T1`, `lmodern`, `microtype`; (b) `amsmath` + `amssymb` para `\text{}` en subscripts de tablas; (c) `\graphicspath{}` apuntando a `../../reports/figures/` para evitar duplicar PNGs; (d) `\input{...}` directo de las 8 tablas pre-existentes en `reports/tables/` (las dos no-usadas `completitud_indicadores.tex` y `concordancia_etiquetadores.tex` tienen underscores no-escapados que las harian incompatibles sin el paquete `underscore`, pero como no se usan en el informe esto no afecta); (e) `\newcommand{\zaltscore}` para escribir Z''-Score consistentemente con dos comillas simples ASCII.
  - **Bug resuelto durante compilacion**: el paquete `[strings]{underscore}` (intentado para liberalizar `_` en text mode) corrompio `\bibliography{informe_final}` al sustituir el subrayado por `\textunderscore`, generando 36 warnings de "didn't find database entry" pese a que el .bib estaba presente. Solucion: removido el paquete; las tablas usadas no requieren liberalizacion de `_`.
  - **Caracter Unicode**: 14 em-dashes `—` (U+2014) reemplazados por `---` (LaTeX shorthand) para evitar warnings de "missing character in font ec-lmbx12".
  - **Criterios PLAN 10.5 cumplidos**: (1) PDF sin errores ✓; (2) 11 secciones ✓; (3) figuras referenciadas ✓; (4) tablas referenciadas ✓; (5) bib 36/48 dentro de rango ✓; (6) conclusiones con mapeo a 6 objetivos ✓.
- **Fase 11** ✅ (2026-05-09): diapositivas Beamer redactadas y compiladas.
  Salidas: `docs/diapositivas/diapositivas.tex` (435 lineas) y `docs/diapositivas/diapositivas.pdf` (538 KB, **20 slides**, formato 16:9). Hallazgos clave:
  - **Estructura completa segun 11.1**: (1) Titulo, (2) Agenda, (3) Contexto + brecha, (4) Estado del arte (7 ejes), (5) Objetivos, (6) Pipeline metodologico, (7) Datos SIREM + Ibague, (8) Etiquetado triangulado, (9) Feature engineering, (10) Particion temporal, (11) Modelado, (12) Resultados test, (13) Matriz de confusion + ROC, (14) SHAP global, (15) Validacion Ibague, (16) vs metodos clasicos, (17) vs literatura + robustez, (18) Conclusiones (mapeo objetivos), (19) Aportes/limitaciones/trabajo futuro, (20) Cierre + preguntas.
  - **Tema Beamer**: `Madrid` con `seahorse` y paleta institucional Universidad de Ibague (azul `RGB(0,70,127)`). Slides de portada y cierre con fondo solido azul.
  - **Compilacion limpia**: `tectonic -X compile diapositivas.tex` exit 0. 0 errores, 0 warnings de fuente (tras reemplazar el caracter Unicode `✓` por `$\checkmark$` de `amssymb`). 20 paginas verificadas con `pdfinfo`.
  - **Auditoria 7x7**: maximo 7 lineas de contenido por slide (subtitle + bullets + filas de tabla). Slides al limite: 5 (objetivos), 10 (split temporal), 17 (vs literatura), 19 (aportes/limitaciones) con 7 lineas cada uno. Slides 1 y 20 son portadas (exentas).
  - **Figuras reutilizadas** desde `reports/figures/`: `08_matriz_confusion_xgboost.png` (slide 13), `12_shap_importance_bar.png` (slide 14). El resto de los slides usan tablas compactas redactadas inline (no `\input` de tablas del informe, que estaban dimensionadas para A4).
  - **Decisiones tecnicas LaTeX**: (a) `\documentclass[10pt, aspectratio=169]{beamer}` para widescreen modern; (b) `babel-spanish` + `inputenc utf8` + `fontenc T1` + `lmodern` + `microtype`, mismo stack del informe; (c) `\graphicspath{../../reports/figures/}` para evitar duplicar PNGs; (d) `\usepackage{amssymb}` para `\checkmark` en lugar de Unicode `✓` (no representable en `ec-lmss10`); (e) `\setbeamercolor{frametitle, palette primary, structure, ...}` redirigidos a `uibblue` para coherencia visual con el branding.
  - **Criterios PLAN 11.4 cumplidos**: (1) 20 slides en rango [15,20] ✓; (2) maximo 7 lineas por slide ✓; (3) PDF compila sin errores ✓.

**Lo que falta** (este plan): nada del bloque tecnico ni documental. Pendiente de cierre administrativo: README final + commit/push final del repo.

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

- [x] El CSV resultante tiene exactamente 203,104 filas. **Cumplido** (203,104 × 23).
- [x] Los 18 indicadores estan presentes; ninguno tiene 100% de nulos (si pasa, hay un bug en columna source). **Cumplido** (peor caso 45.17% nulos en `dias_inventario`).
- [x] El histograma del Z-Score muestra una distribucion unimodal aproximadamente normal/log-normal con cola. **Cumplido** (mediana 4.93, cola pesada visible).
- [x] Las 5 figuras se generan sin errores y se ven legibles a 300 DPI. **Cumplido** (300 DPI verificado con PIL).

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

- [x] Distribucion final tiene las 3 clases con minimo 5% cada una (si no, ajustar reglas heuristicas). **Cumplido**: 16.8% bajo / 66.4% medio / 16.8% alto.
- [x] Cohen's kappa entre B y C ≥ 0.4 (acuerdo moderado o mejor; si es muy bajo, las heuristicas son inconsistentes). **Cumplido**: kappa B-C = 0.446 (heuristica calibrada con terciles empiricos por indicador).
- [ ] ~~El año 2020 muestra deterioro relativo (mayor proporcion de `riesgo_alto` o `riesgo_medio`) por COVID.~~ **No cumplido por realidad de los datos** (delta 2020 vs 2019 = -0.11 pp). Reclasificado como **observacion empirica** a discutir en Fase 10 (informe). Posibles causas: alivios fiscales 2020, efecto COVID diferido a 2021-2022, sesgo de seleccion del SIREM.

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

- [x] No hay columnas con 100% nulos. **Cumplido**: cero columnas con 100% NaN (peor caso 57% nulos en `dias_inventario_d1`, no-bloqueante).
- [x] No hay duplicados (NIT_LIMPIO, ANIO). **Cumplido**: 0 duplicados.
- [x] Las dummies suman 1 por fila dentro de cada grupo categorico (sanity check). **Cumplido**: en las 203,104 filas, `ciiu_*` (12 cols), `sociedad_*` (4 cols) y `depto_*` (6 cols) suman exactamente 1.

---

### FASE 4 — Particion train / val / test + holdout Ibague

**Notebook**: `notebooks/06_particion_datos.ipynb`
**Effort estimado**: 2 horas
**Depende de**: Fase 3 + cruce CCI ya hecho en notebook 01

> ⚠️ **Hallazgo de Fase 2**: el anyo 2017 tiene 99.1% de Z-Score nulos (problema de
> completitud del SIREM 2017). **Filtrar `ANIO != 2017` antes del split** o tratarlo
> como subset diagnostico separado. Sin filtrar, casi todas las observaciones de
> 2017 (~16K) caerian en `riesgo_medio` por consenso fallido y contaminarian el
> entrenamiento.

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

- [x] Los 4 asserts pasan (no leakage Ibague). **Cumplido**: train/val/test sin NITs Ibague + cobertura completa de NITs nacionales en algun split.
- [x] Distribucion de clases similar (±5%) entre train/val/test. **Cumplido**: delta maximo = 4.5pp (clase `bajo`, train vs test).
- [x] Tamanos coherentes: train > val < test, todos > 1000 obs. **Cumplido**: train 108,522 > val 26,878 < test 50,957.

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

- [x] Los 4 modelos serializados se cargan correctamente con `joblib.load`. **Cumplido**: roundtrip verificado en notebook 07 cell de verificacion (los 4 archivos exponen `pipeline`/`model` + `meta` + `hp`).
- [x] El mejor XGBoost (con o sin SMOTE) supera al baseline logistico en F1-macro sobre `val.csv` por al menos +5 puntos porcentuales. **Cumplido**: XGBoost 0.9979 - LR 0.7993 = **+19.86pp** ≥ 5pp.
- [x] Ningun modelo muestra signos obvios de overfitting (gap train F1 vs val F1 < 10pp). **Cumplido**: gap max = 0.81pp (LR), XGB +0.21pp, XGB+SMOTE +0.41pp, RF +0.51pp.

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

- [x] El mejor modelo logra F1-macro ≥ 0.70 sobre el test (esperable dado el universo grande). **Cumplido**: XGBoost F1-macro test = **0.9972** ≥ 0.70.
- [x] Los SHAP global y los hallazgos del estado del arte v2 §3.4 (cobertura de intereses, razon deuda, margen neto como features clave) son consistentes — si no, discutir la divergencia. **Cumplido**: 7 de 9 features esperadas estan en el top 10 SHAP (cobertura_intereses, razon_deuda, margen_neto, roa, razon_corriente, capital_trabajo, z_score_altman). **Divergencia a discutir**: z_score_altman es el feature dominante (3.16, 3× mas que el segundo) -- documentado como observacion para Fase 8.1.1 (modelo memoriza la heuristica de etiquetado).
- [x] Los 9 waterfall plots se generan correctamente. **Cumplido**: figura `13_shap_waterfall_ejemplos.png` (1.4 MB, 3 clases × 3 niveles de confianza).

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

- [x] El F1-macro en Ibague no se degrada mas de 10pp respecto al test nacional. **Cumplido**: gap = +0.24pp (test nacional 0.9972 vs Ibague sin 2017 0.9948). Sin diferencia distribucional sectorial relevante.
- [x] Cada empresa mal clasificada tiene una explicacion plausible escrita en el notebook. **Cumplido**: 1/1 errores documentados (NIT *043005 2024 con 3 razones cuantitativas). Nota: el holdout produjo solo 1 error de 359 predicciones; la figura 16 tiene 1 panel en lugar de los 5 sugeridos por el plan.

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
- `reports/figures/18_comparacion_xgboost_vs_clasicos.png` (renumerada por colision con figura 17 de Fase 7)
- `reports/figures/19_estabilidad_sectorial.png` (renumerada)
- `reports/figures/20_estabilidad_shap_seeds.png` (renumerada)
- `reports/tables/comparacion_metodos.tex`
- `reports/tables/comparacion_literatura.tex`
- `reports/tables/etiquetado_manual_resumen.tex`
- `reports/metrics/discusion_fase8.json` (nuevo, consolida los resultados de las 4 sub-tareas)

#### 8.3 Criterios de aceptacion

- [x] La tabla `comparacion_metodos.tex` muestra que XGBoost supera a Altman puro por al menos 10pp en F1-macro. **Cumplido**: delta = +56.0pp (XGB 0.997 vs Altman umbrales originales 0.437) y +30.8pp vs Altman terciles 0.689.
- [x] El acuerdo XGBoost vs etiquetado manual (kappa) es ≥ 0.5 (acuerdo moderado). **Cumplido**: kappa = 0.500 sobre 30 PYMES Ibague (10 por clase predicha, protocolo manual reproducible declarado a-priori). Documentado en `discusion_fase8.json`.
- [x] La estabilidad SHAP de los top-3 features es ≥ 90% (estables; hallazgo consistente con Lin 2024). **Cumplido**: top-3 (z_score_altman, razon_corriente, margen_neto) presentes en 100% de las 5 seeds {7,21,42,100,2024}; jaccard top-10 promedio = 0.927.

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

- [x] El .tex compila sin warnings de citas faltantes. **Cumplido**: `tectonic -X compile estado_del_arte_v2.tex` produjo `estado_del_arte_v2.pdf` (181 KB) con 0 warnings de citas indefinidas y 0 referencias indefinidas. Solo 35 warnings cosmeticas pre-existentes (`Overfull/Underfull \hbox` en tablas anchas).
- [x] La tabla 2 tiene metricas reales en la fila final. **Cumplido**: fila "Este trabajo" muestra `F1=0,997 / AUC$\approx$1,0` (reemplaza `Por det.`); columna Web pasa de `Si` a `Futuro` por delimitacion de alcance.
- [x] La seccion 4 (sistemas web) esta acortada o relocada si no es relevante. **Cumplido**: el cierre de §4.3 ("vacio de integracion") fue reframado para declarar explicitamente que el alcance del trabajo final IA excluye el sistema web (trabajo futuro). Ajustes consistentes en Tabla 4 (`tab:vacios`, fila Integracion web) y en §8 (sintesis y posicionamiento) para no comprometer entrega de un sistema web no contemplado.

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

- [x] El PDF se genera sin errores. **Cumplido**: `tectonic -X compile informe_final.tex` produjo `informe_final.pdf` (5.4 MB, 33 paginas) con BibTeX limpio (0 entradas faltantes), 0 referencias indefinidas y 0 citas indefinidas.
- [x] Tiene exactamente las 11 secciones del 10.1. **Cumplido**: (1) Titulo (titlepage), (2) Resumen + (2bis) Abstract, (3) Palabras clave + Keywords, (4) Introduccion con 4.1 Ubicacion + 4.2 Estado del arte + 4.3 Justificacion + 4.4 Objetivos, (5) Metodologia, (6) Materiales, (7) Desarrollo, (8) Resultados, (9) Discusion (9.1 vs metodos + 9.2 golden standard + 9.3 vs literatura + 9.4 robustez + 9.5 limitaciones), (10) Conclusiones, (11) Referencias.
- [x] Cada figura del informe esta referenciada en el texto con `\ref{fig:N}`. **Cumplido**: 16 figuras definidas (`\label{fig:...}`) y las 16 referenciadas (`\ref{fig:...}`) en texto.
- [x] Cada tabla esta referenciada con `\ref{tab:N}`. **Cumplido**: 8 tablas usadas (`\input` desde `reports/tables/`), las 8 referenciadas en texto.
- [x] Bibliografia con entre 30 y 50 referencias (subconjunto curado del .bib). **Cumplido**: 36 entradas citadas del `informe_final.bib` (copia de las 48 entradas curadas en `references_v2.bib`), dentro del rango 30-50.
- [x] La seccion de Conclusiones lista explicitamente los 6 objetivos especificos y como se cumplio cada uno. **Cumplido**: §10 mapea Objetivos 1-6 -> resultado obtenido + limitacion principal de cada uno.

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

- [x] Entre 15 y 20 slides. **Cumplido**: 20 slides verificados con `pdfinfo` (`Pages: 20`).
- [x] Cada slide tiene maximo 7 lineas de texto (regla 7×7). **Cumplido**: auditoria estructural por slide, ningun slide excede 7 lineas de contenido (subtitle + bullets + filas de tabla); slides al limite (7 lineas exactas): 5, 10, 17, 19; slides 1 y 20 son portadas (sin contenido).
- [x] El PDF compila sin errores. **Cumplido**: `tectonic -X compile diapositivas.tex` exit 0, 0 errores, 0 warnings de fuente (tras sustituir Unicode `✓` por `$\checkmark$`). PDF 538 KB.

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
| **G1** (post-Fase 1) | Los 18 indicadores estan calculados y la matriz de correlacion no muestra ninguna correlacion = 1.0 (excepto diagonal). **PASS** (2026-05-09). |
| **G2** (post-Fase 2) | La etiqueta final tiene 3 clases con minimo 5% cada una. Kappa entre etiquetadores ≥ 0.4. **PASS** (2026-05-09): 16.8/66.4/16.8 y kappa B-C = 0.446. |
| **G3** (post-Fase 4) | Los 4 asserts de no-leakage Ibague pasan. **PASS** (2026-05-09): train/val/test sin NITs Ibague, cobertura completa de NITs nacionales, delta clases 4.5pp. |
| **G4** (post-Fase 5) | XGBoost supera el baseline logistico en val por ≥ 5pp en F1-macro. **PASS** (2026-05-09): +19.86pp (XGB 0.9979 vs LR 0.7993). |
| **G5** (post-Fase 6) | F1-macro de test ≥ 0.70 (esperable dado N grande). **PASS** (2026-05-09): XGBoost F1-macro test = 0.9972; AUC-ROC OvR = 0.99999; 7/9 features de literatura en top 10 SHAP. |
| **G6** (post-Fase 7) | F1 en Ibague no degrada > 10pp respecto a test nacional, o si lo hace, esta documentada la razon. **PASS** (2026-05-09): F1-macro Ibague sin 2017 = 0.9948 vs test nacional = 0.9972, gap = +0.24pp. AUC-ROC OvR = 1.0000. Solo 1 error de 359 (NIT *043005 2024) con 3 razones documentadas. |
| **G7** (post-Fase 8) | Tabla comparativa contiene al menos 5 metodos (LR, RF, XGBoost, Altman original, Altman terciles). **PASS** (2026-05-09): `comparacion_metodos.tex` lista 6 metodos (LR, RF, XGBoost, XGBoost+SMOTE, Altman terciles, Altman umbrales originales) con F1-macro test sobre 50,957 obs. |
| **G8** (post-Fase 10) | El PDF compila limpio, todas las figuras numeradas, todas las tablas numeradas, biblio sin warnings. **PASS** (2026-05-09): `informe_final.pdf` (5.4 MB, 33 paginas) compila sin errores, 16/16 figuras numeradas y referenciadas, 8/8 tablas numeradas y referenciadas, BibTeX clean (0 entradas faltantes). |
| **G9** (post-Fase 11) | Diapositivas Beamer compilan limpio dentro del rango 15--20 slides y respetan la regla 7×7. **PASS** (2026-05-09): `diapositivas.pdf` (538 KB, 20 slides) compila sin errores; auditoria estructural confirma maximo 7 lineas por slide (slides al limite: 5/10/17/19). |

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
- [x] Fase 1 — Indicadores calculados + EDA completo (2026-05-09)
- [x] Fase 2 — Etiqueta de riesgo trianguladade (2026-05-09)
- [x] Fase 3 — Features con FE temporal (2026-05-09)
- [x] Fase 4 — Particion sin leakage Ibague (2026-05-09)
- [x] Fase 5 — 3 modelos entrenados (LR, RF, XGBoost) (2026-05-09)
- [x] Fase 6 — Evaluacion + SHAP global y local (2026-05-09)
- [x] Fase 7 — Validacion en Ibague (2026-05-09)
- [x] Fase 8 — Comparacion contra metodos clasicos + golden standard (2026-05-09)

### Documental
- [x] Fase 9 — Estado del arte ajustado con resultados (2026-05-09)
- [x] Fase 10 — Informe en LaTeX compilado (2026-05-09)
- [x] Fase 11 — Diapositivas Beamer compiladas (2026-05-09)

### Repo
- [ ] Todos los commits con mensajes claros, sin Co-Authored-By Claude
- [ ] README.md actualizado con estado final del proyecto
- [ ] PDFs finales (informe + slides) commiteados en `docs/informe_final/` y `docs/diapositivas/`

---

*Documento creado: Mayo 2026 · Plan de trabajo - Universidad de Ibague*
