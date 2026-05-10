# Prediccion de Riesgo Financiero en PYMES Colombianas

Trabajo final de la asignatura **Inteligencia Artificial** (Universidad de Ibague). Modelo de Machine Learning supervisado que clasifica el nivel de riesgo financiero (bajo / medio / alto) de PYMES colombianas a partir de sus estados financieros NIIF, usando datos publicos del SIREM (Superintendencia de Sociedades) entre 2016 y 2024, con validacion regional sobre PYMES de Ibague identificadas via cruce con la Camara de Comercio.

## Resultado principal

| Metrica | Test nacional ($n=50\,957$) | Holdout Ibague ($n=359$) |
|---|---|---|
| **F1-macro** | **0,9972** | **0,9948** |
| AUC-ROC OvR macro | 0,99999 | 1,0000 |
| AUC-PR macro | 0,99997 | -- |
| Errores totales | 127 / 50\,957 (0,25%) | 1 / 359 (0,28%) |

XGBoost (`max_depth=8`, `learning_rate=0,1`, `sample_weight=balanced`, `early_stopping_rounds=30`) supera a:

- la regresion logistica baseline por **+18,7 pp** en F1-macro de test,
- la heuristica de Altman (terciles empiricos) por **+30,8 pp**,
- la heuristica de Altman (umbrales originales 1,1 / 2,6) por **+56,0 pp**.

Las features mas importantes segun TreeSHAP son `z_score_altman`, `margen_neto`, `razon_corriente`, `cobertura_intereses` y `razon_deuda`.

## Documentos entregables

| Documento | Ruta | Contenido |
|---|---|---|
| Informe final | [`docs/informe_final/informe_final.pdf`](docs/informe_final/informe_final.pdf) | 33 paginas, 11 secciones, 36 referencias citadas |
| Diapositivas | [`docs/diapositivas/diapositivas.pdf`](docs/diapositivas/diapositivas.pdf) | 20 slides Beamer (16:9, tema Madrid) |
| Estado del arte | [`docs/literatura/Estado del arte/latex/estado_del_arte_v2.pdf`](docs/literatura/Estado%20del%20arte/latex/estado_del_arte_v2.pdf) | 48 referencias, 7 ejes tematicos |
| Plan operativo | [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md) | 11 fases con criterios de aceptacion, dependencias, pitfalls |
| Diccionario de datos | [`docs/DICCIONARIO_DATOS_ML.md`](docs/DICCIONARIO_DATOS_ML.md) | Definicion de los 18 indicadores y sus formulas |
| Reporte ETL | [`reports/REPORTE_ETL_NACIONAL.md`](reports/REPORTE_ETL_NACIONAL.md) | Detalle del ETL nacional y anomalias detectadas |

## Pipeline ejecutado

Diez notebooks numerados producen un dataset reproducible y un modelo evaluado:

```
01_etl_camara_comercio.ipynb        Cruce CCI Ibague <-> SIREM (61 NITs)
02_etl_nacional_pymes.ipynb         Consolidacion de los 4 estados financieros
                                    -> data/processed/colombia_consolidado_pymes.csv
03_indicadores_financieros.ipynb    18 indicadores + Z''-Score Altman
04_etiquetado_riesgo.ipynb          Etiqueta triangulada (Z'' + heuristica + cuartiles)
05_feature_engineering.ipynb        71 features (deltas, crecimientos, dummies, escala)
06_particion_datos.ipynb            Split temporal + holdout Ibague
07_modelado.ipynb                   LR / RF / XGBoost / XGBoost+SMOTE
08_evaluacion_shap.ipynb            Test, SHAP global y local
09_validacion_ibague.ipynb          Aplicacion al holdout regional
10_discusion_comparativa.ipynb      vs Altman, vs etiquetado manual, robustez
```

### Convencion train / validation / test

- **Train**: 2016, 2018-2021 (108\,522 obs.)
- **Validation**: 2022 (26\,878 obs.)
- **Test**: 2023-2024 (50\,957 obs.)
- **Holdout Ibague**: 61 PYMES, 359 obs. (incluye 2017 para diagnostico)

El ano 2017 se excluye del split nacional debido a una anomalia de captura del SIREM (99,1% de Z-Score nulos). Los 61 NITs de Ibague nunca aparecen en train / val / test (cuatro asserts de no-leakage verificados).

## Estructura del repositorio

```
.
|-- data/
|   |-- processed/                     CSVs consolidados (LFS-tracked)
|   |   |-- colombia_consolidado_pymes.csv     203\,104 x 230 -- salida del ETL
|   |   |-- colombia_indicadores_pymes.csv     203\,104 x 23  -- 18 indicadores + Z''
|   |   |-- colombia_etiquetas_riesgo.csv      203\,104 x 9   -- etiqueta triangulada
|   |   |-- colombia_features_ml.csv           203\,104 x 77  -- features ML
|   |   `-- nacional_{train,val,test}.csv      splits temporales sin Ibague
|   `-- ibague/                        holdout regional + predicciones
|-- notebooks/                         pipeline numerado 01-10
|-- src/
|   |-- indicadores.py                 18 indicadores + Z''-Score
|   `-- etl_utils.py                   carga del consolidado + normalizacion mojibake
|-- scripts/                           generadores nbformat de notebooks
|-- reports/
|   |-- figures/                       20 figuras a 300 DPI
|   |-- tables/                        10 tablas LaTeX
|   |-- metrics/                       7 JSONs con metricas y resumenes
|   `-- REPORTE_ETL_NACIONAL.md
|-- docs/
|   |-- informe_final/                 .tex + .bib + .pdf (33 pp)
|   |-- diapositivas/                  .tex + .pdf (20 slides)
|   |-- literatura/                    estado del arte + bibliografia
|   |-- DICCIONARIO_DATOS_ML.md
|   |-- DATASET_SOURCES.md
|   `-- GUIA_ESTADO_DEL_ARTE.md
|-- models/                            (gitignored) joblib serializados
|-- PLAN_DE_TRABAJO.md
|-- requirements.txt
`-- README.md
```

## Datos

Los CSVs procesados se versionan via Git LFS (`.gitattributes` declara `*.csv filter=lfs`):

```bash
git lfs install
git lfs pull
```

Los CSVs crudos del SIREM (~9,17 GB, 4 archivos) **no son necesarios** para reproducir el modelo; la consolidacion ya esta en `data/processed/`. Si se requiere re-correr el ETL desde fuente cruda, ver [`docs/DATASET_SOURCES.md`](docs/DATASET_SOURCES.md) para los enlaces de descarga.

> **Nota sobre encoding**: el consolidado tiene mojibake (`ï¿½` por vocales tildadas) en 105 nombres de columna. **Cargar siempre via** `src.etl_utils.cargar_consolidado()` -- aplica `normalizar_columnas()` y entrega nombres ASCII limpios. Las constantes en `src/indicadores.py` asumen ya esa normalizacion.

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Linux/Mac
.\venv\Scripts\activate             # Windows

pip install -r requirements.txt
jupyter notebook
```

Dependencias clave: `pandas`, `numpy`, `scikit-learn`, `xgboost` (con soporte GPU opcional via CUDA), `shap`, `imbalanced-learn`, `matplotlib`, `seaborn`. Para Python 3.14 se recomienda usar versiones `>=` sin pinear (las versiones de `requirements.txt` apuntan al perfil 2.x / 1.x para compatibilidad amplia).

Para compilar los documentos LaTeX:

```bash
tectonic -X compile docs/informe_final/informe_final.tex
tectonic -X compile docs/diapositivas/diapositivas.tex
tectonic -X compile "docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex"
```

## Como reproducir

1. Clonar el repositorio y traer los CSVs LFS (`git lfs pull`).
2. Crear el venv e instalar dependencias.
3. Ejecutar los notebooks en orden, de `03` a `10`. Los notebooks `01` y `02` corresponden al ETL desde los datasets crudos del SIREM y solo son necesarios si se quiere re-construir el consolidado.
4. Cada notebook deja sus salidas en `data/processed/`, `data/ibague/`, `reports/figures/`, `reports/tables/` y `reports/metrics/`.

Convenciones (ver [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md) §2 para el detalle):

- `random_state=42` y `np.random.seed(42)` en todas las celdas iniciales.
- Holdout Ibague (61 NITs) prohibido en train / val / test nacional (asserts en notebook 06).
- Split temporal: train 2016-2021, val 2022, test 2023-2024.
- Salidas estandarizadas en `data/processed/`, `reports/figures/`, `reports/tables/`, `reports/metrics/`.

## Decisiones metodologicas

- **Etiquetado**: Z''-Score Altman para mercados emergentes con terciles empiricos del dataset (no umbrales originales) + heuristica de 5+5 senales con terciles por indicador. Etiqueta final por consenso B$\equiv$C; Cohen $\kappa_{B,C} = 0{,}446$.
- **Modelo principal**: XGBoost con `sample_weight` balanceado -- mejor F1-macro en validacion (0,9979) y test (0,9972).
- **Baselines**: regresion logistica + Random Forest + XGBoost + SMOTE.
- **Validacion temporal**: cronologica estricta, sin K-Fold aleatorio que introduciria leakage por empresa.
- **Interpretabilidad**: TreeSHAP global (5\,000 obs estratificadas) + 9 waterfall plots locales.
- **Robustez verificada**: estabilidad temporal (gap 0,03 pp), sectorial (gap 0,22 pp), SHAP top-3 al 100% sobre 5 seeds.

## Estado del proyecto

Las once fases del [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md) estan ejecutadas:

| Fase | Resultado |
|---|---|
| 1. Indicadores + EDA | 18 indicadores + Z''-Score, 5 figuras |
| 2. Etiquetado triangulado | $\kappa_{B,C}=0{,}446$, distribucion 16,8 / 66,4 / 16,8 |
| 3. Feature engineering | 71 features (7 familias) winsorizadas p1-p99 |
| 4. Particion temporal | 4 asserts de no-leakage Ibague PASS |
| 5. Modelado | 4 modelos, 70 fits, GPU NVIDIA RTX 2060 |
| 6. Evaluacion + SHAP | F1-macro test 0,9972; 7/9 features de literatura en top 10 |
| 7. Validacion Ibague | F1-macro 0,9948 (sin 2017), 1 error en 359 |
| 8. Discusion | $+56$ pp vs Altman puro; kappa manual = 0,500 |
| 9. Estado del arte v2 | 48 referencias, 7 ejes, ajustado con resultados |
| 10. Informe final | 33 paginas LaTeX, 16 figuras, 8 tablas, 36 citas |
| 11. Diapositivas | 20 slides Beamer, regla 7x7 cumplida |

## Autores

Juan Camilo Perea  *  German
Universidad de Ibague -- Facultad de Ingenieria
2026
