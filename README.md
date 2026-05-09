# Analisis de Riesgo Financiero en PYMES — Trabajo Final IA

Trabajo final de la asignatura **Inteligencia Artificial**. Construimos un modelo de Machine Learning para clasificar el riesgo financiero de PYMES colombianas a partir de estados financieros publicos del SIREM (Superintendencia de Sociedades).

## Convencion de entrenamiento y validacion

- **Entrenamiento y desarrollo**: dataset **nacional** (38,245 PYMES NIIF Pymes, 2016–2024, 203,104 observaciones empresa-ano).
- **Validacion / caso de estudio**: subconjunto de **Ibague** (61 PYMES identificadas en SIREM via cruce con la Camara de Comercio).

Esta separacion nacional → Ibague evita data leakage: el modelo no ve a las empresas de Ibague durante el entrenamiento y se reserva como caso de prueba final.

## Estructura del repositorio

```
.
├── data/
│   ├── processed/   CSVs consolidados generados por el ETL (LFS-tracked)
│   └── ibague/      Subconjunto de PYMES de Ibague para validacion
├── notebooks/
│   ├── 01_etl_camara_comercio.ipynb   ETL Camara de Comercio + cruce SIREM
│   └── 02_etl_nacional_pymes.ipynb    ETL nacional NIIF Pymes (consolidacion)
├── src/
│   ├── indicadores.py   18 indicadores financieros + Z''-Score Altman
│   ├── etl_utils.py     Carga del consolidado + normalizacion de mojibake
│   └── __init__.py
├── reports/
│   ├── REPORTE_ETL_NACIONAL.md   Reporte detallado del ETL ya ejecutado
│   └── figures/                   Figuras generadas por los notebooks
├── docs/
│   ├── DICCIONARIO_DATOS_ML.md   Diccionario de datos e indicadores
│   ├── DATASET_SOURCES.md        Origen y descarga de los CSV crudos
│   ├── GUIA_ESTADO_DEL_ARTE.md   Guia metodologica del estado del arte
│   └── literatura/
│       ├── Estado del arte/      LaTeX, .bib y PDFs por eje tematico
│       ├── PROPUESTA_FINAL_Trabajo_Grado.pdf
│       └── GUIA_COMPLETA_Investigacion_Primera_Tesis.pdf
├── models/          (gitignored) Modelos entrenados
├── requirements.txt
├── README.md
├── CLAUDE.md
├── .gitignore
└── .gitattributes   Reglas Git LFS para los CSVs procesados
```

## Datos

| Dataset | Origen | Tamano | Necesidad | Estado |
|---|---|---|---|---|
| 4 CSVs SIREM crudos | Supersociedades | ~9.17 GB | **No requerido** para este trabajo | gitignored |
| Camara de Comercio Ibague | CCI | ~25 MB | **No requerido** para este trabajo | gitignored |
| 5 CSVs procesados | Notebook 02 | ~367 MB | **Si requerido** | en `data/processed/` |

Los **datos crudos no son necesarios** para este trabajo final de IA porque el ETL ya fue ejecutado y los archivos consolidados estan disponibles. Solo se requeririan si se quisiera re-correr el ETL (e.g. nuevos cortes del SIREM).

### Como conseguir los CSVs procesados

Los archivos viven como punteros Git LFS en el repositorio principal `IngRetrius/Tesis`. **No se necesita instalar `git-lfs`**: GitHub sirve los archivos directamente a traves de `media.githubusercontent.com`.

```bash
cd data/processed
for f in colombia_consolidado_pymes.csv colombia_situacion_financiera_pymes.csv \
         colombia_resultado_integral_pymes.csv colombia_flujo_efectivo_pymes.csv \
         colombia_metadata_pymes.csv; do
  curl -L -o "$f" "https://media.githubusercontent.com/media/IngRetrius/Tesis/main/data/$f"
done
```

Verificar tamanos esperados:

| Archivo | Filas (incluye header) | Tamano |
|---|---|---|
| colombia_consolidado_pymes.csv | 203,105 | 214 MB |
| colombia_situacion_financiera_pymes.csv | 202,502 | 63 MB |
| colombia_flujo_efectivo_pymes.csv | 203,105 | 49 MB |
| colombia_resultado_integral_pymes.csv | 203,104 | 27 MB |
| colombia_metadata_pymes.csv | 38,246 | 16 MB |

> Nota sobre encoding: los CSVs tienen mojibake (`ï¿½` por vocales tildadas) en 105 nombres de columna. Usar siempre `src.etl_utils.cargar_consolidado()` o llamar `normalizar_columnas()` despues de `pd.read_csv()` para obtener nombres ASCII limpios. Las constantes en `src/indicadores.py` asumen los nombres ya normalizados.

## Pipeline planeado

```
data crudo (SIREM + CCI)
    │
    ├─ 01_etl_camara_comercio.ipynb     [HECHO]
    │  cruce CCI <-> SIREM, normaliza NIT (descubre logica de DV)
    │
    └─ 02_etl_nacional_pymes.ipynb      [HECHO]
       consolida los 4 estados financieros del SIREM (NIIF Pymes)
       │
       ▼
   data/processed/colombia_consolidado_pymes.csv   (203K obs x 230 cols)

   ┌────────────────────────────────────────────────────────────────┐
   │ Pendientes:                                                    │
   │                                                                │
   │  03  Calculo de los 18 indicadores + Z''-Score Altman          │
   │      (usar src/indicadores.py)                                 │
   │                                                                │
   │  04  Etiquetado de riesgo (bajo/medio/alto)                    │
   │      Z''-Score por terciles + reglas heuristicas como          │
   │      triangulacion (sec. 2.3 del estado del arte v2)           │
   │                                                                │
   │  05  Modelado: Logistica baseline -> Random Forest -> XGBoost  │
   │      Split temporal: train 2016-2021 / val 2022 / test 2023-24 │
   │      SMOTE solo dentro de los pliegues de entrenamiento        │
   │                                                                │
   │  06  Evaluacion + interpretabilidad                            │
   │      F1-macro, AUC-PR, matriz de confusion                     │
   │      TreeSHAP (compatible nativamente con XGBoost)             │
   │                                                                │
   │  07  Validacion sobre PYMES de Ibague                          │
   │      Aplicar el modelo nacional al subconjunto reservado       │
   │      Comparar predicciones con indicadores observados          │
   └────────────────────────────────────────────────────────────────┘
```

## Decisiones metodologicas

Basadas en `docs/literatura/Estado del arte/latex/estado_del_arte_v2.tex`:

- **Etiquetado**: Z''-Score Altman para mercados emergentes con umbrales calibrados por terciles del dataset (no los originales).
- **Modelo principal**: XGBoost — justificado por integracion nativa con TreeSHAP, latencia rapida, paridad de desempeno con LightGBM/CatBoost.
- **Baselines**: Regresion logistica (interpretable) + Random Forest.
- **Desbalance de clases**: SMOTE aplicado solo dentro de pliegues de entrenamiento, comparado contra `scale_pos_weight` nativo de XGBoost.
- **Validacion temporal**: train 2016–2021, val 2022, test 2023–2024 (sin leakage por empresa).
- **Interpretabilidad**: TreeSHAP para explicaciones globales y locales.

## Setup

```bash
# Crear entorno virtual
python -m venv venv

# Activar
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Lanzar Jupyter
jupyter notebook
```

Dependencias clave: `pandas 2.2`, `numpy 1.26`, `scikit-learn 1.4`, `xgboost 2.0`, `shap 0.44`, `imbalanced-learn 0.12`.

## Estado actual

- [x] ETL Camara de Comercio (notebook 01)
- [x] ETL nacional NIIF Pymes (notebook 02)
- [x] Estado del arte v2 redactado (48 referencias)
- [x] Modulo de indicadores financieros (`src/indicadores.py`)
- [x] Modulo de utilidades ETL con normalizacion de mojibake (`src/etl_utils.py`)
- [x] Datos consolidados descargados localmente (5 CSVs, 367 MB)
- [ ] Notebook 03 — calculo de indicadores sobre el dataset nacional
- [ ] Notebook 04 — etiquetado de riesgo
- [ ] Notebook 05 — entrenamiento (LR / RF / XGBoost)
- [ ] Notebook 06 — evaluacion + TreeSHAP
- [ ] Notebook 07 — validacion sobre Ibague

## Plan de trabajo

El roadmap completo del proyecto (11 fases con criterios de aceptacion, dependencias, pitfalls, mapeo a las secciones del informe final y plan de slides) esta en [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md). Es el documento operativo que cualquier integrante (humano o IA) puede ejecutar fase por fase.

## Notas operativas

- **Notebooks 01 y 02 ya fueron ejecutados** y produjeron los CSVs en `data/processed/`. No es necesario re-correrlos para este trabajo final. Si se quisiera, las rutas absolutas Windows (`C:\Users\USUARIO1\Documents\Tesis\dataset`) requieren ajuste en Linux/Mac (celda 3 de cada notebook).
- **Encoding/mojibake**: 105 columnas del consolidado tienen `ï¿½` por vocales tildadas. Cargar siempre via `src.etl_utils.cargar_consolidado()` o llamar `normalizar_columnas()` para obtener nombres limpios.
- **Modelos** (`models/`) estan gitignored: se regeneran corriendo el notebook 05.
- **`src/`** funciona como paquete Python: desde un notebook en `notebooks/` usar `sys.path.insert(0, '..')` o ejecutar Jupyter desde la raiz del repo.

## Autores

Juan Camilo Perea · German
Universidad de Ibague — Ingenieria de Sistemas
