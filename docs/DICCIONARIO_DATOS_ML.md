# Diccionario de Datos e Indicadores Financieros para Machine Learning

## Proyecto

**Titulo**: "Diseno e Implementacion de un Sistema Web de Analisis Financiero con Modelo de Machine Learning para la Evaluacion de Riesgo en PYMES del Municipio de Ibague, Tolima"

---

## 1. Fuentes de Datos

### 1.1 SIREM (Superintendencia de Sociedades)

El **Sistema de Informacion y Reporte Empresarial (SIREM)** es la base de datos publica de la Superintendencia de Sociedades de Colombia. Contiene los estados financieros de las empresas vigiladas, reportados bajo **Normas Internacionales de Informacion Financiera (NIIF)**.

| Caracteristica | Detalle |
|---|---|
| Cobertura | Empresas vigiladas por Supersociedades en Colombia |
| Formato | CSV, formato vertical (una fila por concepto-valor) |
| Normativa | NIIF Pymes (Grupo 2) y NIIF Plenas (Grupo 1) |
| Periodo | 2016 - 2024 |
| Tamano total | ~9.17 GB (4 archivos) |

#### Datasets utilizados:

| Dataset | Descripcion | Tamano | Conceptos |
|---|---|---|---|
| Estado de Situacion Financiera | Balance general (activos, pasivos, patrimonio) | 4.10 GB | ~66 |
| Estado de Resultado Integral | Perdidas y ganancias (ingresos, costos, utilidades) | 1.57 GB | ~19 |
| Estado de Flujo de Efectivo | Movimientos de caja (operacion, inversion, financiacion) | 1.43 GB | ~101 |

> **Nota**: El dataset de Caratula (~2.07 GB) contiene metadatos (razon social, direccion, etc.) y no se utiliza para el analisis financiero. La informacion de las empresas se obtiene de la Camara de Comercio.

### 1.2 Camara de Comercio de Ibague

Registro mercantil de las empresas activas en la jurisdiccion de la Camara de Comercio de Ibague, con corte a diciembre de 2025.

| Caracteristica | Detalle |
|---|---|
| Registros | 20,280 |
| Columnas | 85 |
| Uso en el proyecto | Identificar las PYMES de Ibague y obtener su informacion general |

---

## 2. Variables del Dataset

### 2.1 Variables de Identificacion (Metadatos)

Estas variables provienen de la Camara de Comercio y del proceso de JOIN:

| Variable | Tipo | Descripcion | Ejemplo |
|---|---|---|---|
| `NIT_LIMPIO` | String | NIT base de 9 digitos (sin digito de verificacion) | `890702298` |
| `RAZON SOCIAL` | String | Nombre legal de la empresa | `FERRETERIA GODOY S.A.` |
| `ORGANIZACION` | Categorica | Tipo de sociedad comercial | `S.A.S`, `SOCIEDAD LIMITADA` |
| `CATEGORIA` | Categorica | Categoria de matricula en Camara | `SOCIEDAD` |
| `GRUPO NIIF` | Categorica | Grupo de reporte NIIF | `GRUPO 2` (Pymes) |
| `ANIO` | Entero | Anio fiscal del reporte financiero | `2023` |

### 2.2 Estado de Situacion Financiera (Balance General)

El balance general presenta la **posicion financiera** de la empresa en un momento dado. Muestra lo que la empresa **tiene** (activos), lo que **debe** (pasivos) y lo que le **pertenece a los socios** (patrimonio).

**Ecuacion fundamental**: Activos = Pasivos + Patrimonio

#### Activos (lo que la empresa posee o controla)

| Concepto SIREM | Definicion | Uso en indicadores |
|---|---|---|
| **Total de activos** | Suma de todos los bienes y derechos de la empresa | ROA, Rotacion de activos |
| **Activos corrientes totales** | Activos que se espera convertir en efectivo en menos de 1 anio (caja, cuentas por cobrar, inventarios) | Razon corriente, Prueba acida, Capital de trabajo |
| **Efectivo y equivalentes al efectivo** | Dinero en caja y bancos, e inversiones de alta liquidez | Razon de efectivo |
| **Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes** | Dinero que los clientes deben a la empresa por ventas a credito | Rotacion de cartera |
| **Inventarios corrientes** | Bienes disponibles para la venta o en proceso de produccion | Prueba acida, Rotacion de inventarios |
| **Propiedades, planta y equipo** | Bienes tangibles de uso prolongado (edificios, maquinaria, vehiculos) | Indicador de intensidad de capital |
| **Activos no corrientes totales** | Activos de largo plazo (mas de 1 anio): propiedades, inversiones permanentes | Estructura de activos |
| **Activos biologicos corrientes** | Animales y plantas vivos mantenidos para la venta (aplica a empresas agropecuarias) | Especifico del sector |
| **Otros activos financieros corrientes** | Inversiones financieras de corto plazo | Liquidez ampliada |
| **Activos por impuestos corrientes** | Saldos a favor con la DIAN | Posicion tributaria |

#### Pasivos (lo que la empresa debe)

| Concepto SIREM | Definicion | Uso en indicadores |
|---|---|---|
| **Total pasivos** | Suma de todas las deudas y obligaciones | Razon de deuda, Apalancamiento |
| **Pasivos corrientes totales** | Deudas que vencen en menos de 1 anio | Razon corriente, Capital de trabajo |
| **Pasivos no corrientes totales** | Deudas de largo plazo (mas de 1 anio) | Estructura de deuda |
| **Cuentas comerciales por pagar y otras cuentas por pagar** | Deudas con proveedores por compras a credito | Rotacion de proveedores |
| **Prestamos corrientes** | Creditos bancarios de corto plazo | Estructura de financiamiento |
| **Pasivos por impuestos corrientes** | Impuestos por pagar | Posicion tributaria |
| **Provisiones corrientes por beneficios a los empleados** | Prestaciones sociales pendientes | Obligaciones laborales |

#### Patrimonio (lo que pertenece a los socios)

| Concepto SIREM | Definicion | Uso en indicadores |
|---|---|---|
| **Patrimonio total** | Diferencia entre activos y pasivos = valor neto de la empresa | ROE, Razon deuda/patrimonio |
| **Capital emitido** | Aportes de los socios al capital social | Estructura de capital |
| **Ganancias acumuladas** | Utilidades de periodos anteriores no distribuidas | Politica de retencion |
| **Resultado del periodo** | Utilidad o perdida del anio fiscal actual | Rentabilidad |

### 2.3 Estado de Resultado Integral (P&G)

El estado de resultados muestra el **desempeno** de la empresa durante un periodo: cuanto vendio, cuanto le costo, y cuanto gano o perdio.

| Concepto SIREM | Definicion | Uso en indicadores |
|---|---|---|
| **Ingresos de actividades ordinarias** | Ventas netas por la actividad principal de la empresa | Todos los margenes, Rotacion de activos |
| **Costo de ventas** | Costo directo de producir o adquirir los bienes vendidos | Margen bruto |
| **Ganancia bruta** | Ingresos - Costo de ventas = rentabilidad antes de gastos operativos | Margen bruto |
| **Gastos de ventas** | Gastos relacionados con la comercializacion (publicidad, comisiones, transporte) | Eficiencia comercial |
| **Gastos de administracion** | Gastos de gestion de la empresa (nomina administrativa, arriendo oficinas, servicios) | Eficiencia administrativa |
| **Ganancia (perdida) por actividades de operacion** | Resultado de la operacion principal del negocio, antes de ingresos/gastos financieros | Margen operacional, EBITDA |
| **Ingresos financieros** | Rendimientos de inversiones, intereses ganados | Resultado financiero neto |
| **Costos financieros** | Intereses pagados por prestamos y creditos | Cobertura de intereses |
| **Ganancia (perdida), antes de impuestos** | Resultado antes de pagar impuesto de renta | Carga tributaria |
| **Impuesto a las ganancias** | Impuesto de renta del periodo | Tasa efectiva de impuestos |
| **Ganancia (perdida)** | **Resultado neto final** (utilidad o perdida del ejercicio) | ROA, ROE, Margen neto |

### 2.4 Estado de Flujo de Efectivo

El flujo de efectivo muestra los **movimientos reales de dinero** de la empresa, clasificados en tres actividades:

#### Actividades de Operacion (el negocio principal)

| Concepto SIREM | Definicion |
|---|---|
| **Flujos de efectivo netos procedentes de actividades de operacion** | Efectivo generado (o consumido) por la operacion principal del negocio |
| **(+/-) Ganancia (perdida)** | Punto de partida: resultado neto del periodo |
| **(+) Gastos por depreciacion y amortizacion** | Se suman porque no representan salida real de efectivo |
| **(+/-) Ajustes por deterioro de valor** | Perdidas por deterioro de activos |
| **(+/-) Ajustes por cambios en inventarios** | Variacion del inventario en el periodo |
| **(+/-) Ajustes por cambios en cuentas por cobrar** | Variacion de lo que deben los clientes |
| **(+/-) Ajustes por cambios en cuentas por pagar** | Variacion de lo que se debe a proveedores |

#### Actividades de Inversion (compra/venta de activos de largo plazo)

| Concepto SIREM | Definicion |
|---|---|
| **Flujos de efectivo netos procedentes de actividades de inversion** | Efectivo usado en compra o venta de activos fijos e inversiones |
| **Compras de propiedades, planta y equipo** | Inversiones en activos fijos |

#### Actividades de Financiacion (deuda y capital)

| Concepto SIREM | Definicion |
|---|---|
| **Flujos de efectivo netos procedentes de actividades de financiacion** | Efectivo de prestamos, pagos de deuda, aportes de socios, dividendos |
| **Incremento (disminucion) neto de efectivo** | **Cambio neto total** en la caja de la empresa durante el periodo |

---

## 3. Indicadores Financieros Calculables

Los indicadores financieros se calculan a partir de las variables del dataset consolidado. Cada indicador tiene una **formula**, una **interpretacion** y un **umbral de referencia** para evaluar la salud financiera.

### 3.1 Indicadores de Liquidez

Miden la **capacidad de la empresa para cumplir sus obligaciones de corto plazo**.

| Indicador | Formula | Interpretacion | Referencia |
|---|---|---|---|
| **Razon Corriente** | Activos corrientes / Pasivos corrientes | Cuantos pesos de activo corriente hay por cada peso de pasivo corriente | > 1.5 deseable; < 1.0 riesgo alto |
| **Prueba Acida** | (Activos corrientes - Inventarios) / Pasivos corrientes | Liquidez excluyendo inventarios (que son menos liquidos) | > 1.0 deseable; < 0.5 riesgo alto |
| **Capital de Trabajo** | Activos corrientes - Pasivos corrientes | Excedente (o deficit) de recursos de corto plazo en pesos | > 0 deseable; < 0 riesgo |
| **Razon de Efectivo** | Efectivo / Pasivos corrientes | Capacidad de pago inmediato solo con caja | > 0.2 aceptable |

#### Formulas en Python:
```python
df['razon_corriente'] = df['Activos corrientes totales'] / df['Pasivos corrientes totales']
df['prueba_acida'] = (df['Activos corrientes totales'] - df['Inventarios corrientes']) / df['Pasivos corrientes totales']
df['capital_trabajo'] = df['Activos corrientes totales'] - df['Pasivos corrientes totales']
df['razon_efectivo'] = df['Efectivo y equivalentes al efectivo'] / df['Pasivos corrientes totales']
```

### 3.2 Indicadores de Rentabilidad

Miden la **capacidad de la empresa para generar utilidades** a partir de sus recursos.

| Indicador | Formula | Interpretacion | Referencia |
|---|---|---|---|
| **Margen Bruto** | Ganancia bruta / Ingresos ordinarios | Porcentaje de cada peso de venta que queda despues de cubrir el costo directo | Varia por sector; > 30% es bueno para comercio |
| **Margen Operacional** | Ganancia operacional / Ingresos ordinarios | Porcentaje que queda despues de cubrir costos y gastos operativos | > 10% deseable; < 0% perdida operacional |
| **Margen Neto** | Ganancia (perdida) / Ingresos ordinarios | Porcentaje final de utilidad por cada peso vendido | > 5% deseable; < 0% perdida neta |
| **ROA** (Return on Assets) | Ganancia (perdida) / Total de activos | Rendimiento de cada peso invertido en activos | > 5% bueno; < 0% destruye valor |
| **ROE** (Return on Equity) | Ganancia (perdida) / Patrimonio total | Rendimiento para los socios por cada peso de patrimonio | > 10% bueno; debe superar costo de capital |

#### Formulas en Python:
```python
df['margen_bruto'] = df['Ganancia bruta'] / df['Ingresos de actividades ordinarias']
df['margen_operacional'] = df['Ganancia (perdida) por actividades de operacion'] / df['Ingresos de actividades ordinarias']
df['margen_neto'] = df['Ganancia (perdida)'] / df['Ingresos de actividades ordinarias']
df['roa'] = df['Ganancia (perdida)'] / df['Total de activos']
df['roe'] = df['Ganancia (perdida)'] / df['Patrimonio total']
```

### 3.3 Indicadores de Endeudamiento (Solvencia)

Miden la **estructura de financiamiento** y el **nivel de deuda** de la empresa.

| Indicador | Formula | Interpretacion | Referencia |
|---|---|---|---|
| **Razon de Deuda** | Total pasivos / Total de activos | Proporcion de activos financiados con deuda | < 0.6 (60%) deseable; > 0.8 riesgo alto |
| **Razon Deuda/Patrimonio** | Total pasivos / Patrimonio total | Cuantos pesos de deuda hay por cada peso de patrimonio | < 2.0 deseable; > 3.0 muy apalancada |
| **Apalancamiento Financiero** | Total de activos / Patrimonio total | Multiplicador de los recursos propios | 1.0-3.0 normal; > 5.0 alto riesgo |
| **Cobertura de Intereses** | Ganancia operacional / Costos financieros | Cuantas veces la utilidad operacional cubre los intereses | > 3.0 deseable; < 1.0 no cubre intereses |

#### Formulas en Python:
```python
df['razon_deuda'] = df['Total pasivos'] / df['Total de activos']
df['deuda_patrimonio'] = df['Total pasivos'] / df['Patrimonio total']
df['apalancamiento'] = df['Total de activos'] / df['Patrimonio total']
df['cobertura_intereses'] = df['Ganancia (perdida) por actividades de operacion'] / df['Costos financieros']
```

### 3.4 Indicadores de Eficiencia (Actividad)

Miden la **velocidad con que la empresa convierte sus recursos en ventas o efectivo**.

| Indicador | Formula | Interpretacion | Referencia |
|---|---|---|---|
| **Rotacion de Activos** | Ingresos ordinarios / Total de activos | Cuantos pesos de venta genera cada peso de activo | > 1.0 eficiente; varia por sector |
| **Rotacion de Inventarios** | Costo de ventas / Inventarios corrientes | Cuantas veces se renueva el inventario al anio | Depende del sector; mas alto = mas eficiente |
| **Rotacion de Cartera** | Ingresos ordinarios / Cuentas por cobrar corrientes | Cuantas veces se cobra la cartera al anio | Mas alto = cobra mas rapido |
| **Dias de Inventario** | 365 / Rotacion de inventarios | Cuantos dias tarda en vender su inventario | < 60 dias deseable para comercio |
| **Dias de Cartera** | 365 / Rotacion de cartera | Cuantos dias tarda en cobrar a sus clientes | < 60 dias deseable |

#### Formulas en Python:
```python
df['rotacion_activos'] = df['Ingresos de actividades ordinarias'] / df['Total de activos']
df['rotacion_inventarios'] = df['Costo de ventas'] / df['Inventarios corrientes']
df['rotacion_cartera'] = df['Ingresos de actividades ordinarias'] / df['Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes']
df['dias_inventario'] = 365 / df['rotacion_inventarios']
df['dias_cartera'] = 365 / df['rotacion_cartera']
```

### 3.5 Resumen de Indicadores

| Categoria | Indicadores | Cantidad |
|---|---|---|
| Liquidez | Razon corriente, Prueba acida, Capital de trabajo, Razon de efectivo | 4 |
| Rentabilidad | Margen bruto, Margen operacional, Margen neto, ROA, ROE | 5 |
| Endeudamiento | Razon deuda, D/P, Apalancamiento, Cobertura de intereses | 4 |
| Eficiencia | Rotacion de activos, inventarios, cartera, Dias de inventario, Dias de cartera | 5 |
| **Total** | | **18 indicadores** |

---

## 4. Aplicacion en Machine Learning

### 4.1 Definicion del Problema

**Objetivo**: Construir un modelo de Machine Learning que evalue el **nivel de riesgo financiero** de las PYMES de Ibague, clasificandolas en categorias de riesgo (bajo, medio, alto) a partir de sus indicadores financieros.

**Tipo de problema**: Clasificacion supervisada (multiclase)

**Alcance**: El modelo se entrenara con datos historicos (2016-2024) de las PYMES identificadas en el SIREM, y se integrara en un sistema web que permita a los usuarios consultar el riesgo de una empresa o analizar tendencias del sector.

### 4.2 Variable Objetivo (Target): Etiquetas de Riesgo

La variable objetivo no existe directamente en los datos. Debe ser **construida** a partir de los indicadores financieros. Existen varias estrategias:

#### Estrategia 1: Z-Score de Altman (adaptado para PYMES colombianas)

El modelo Z de Altman es un modelo clasico de prediccion de quiebra. La version para empresas no cotizadas en bolsa es:

```
Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
```

Donde:
- **X1** = Capital de trabajo / Total de activos (liquidez)
- **X2** = Ganancias acumuladas / Total de activos (rentabilidad acumulada)
- **X3** = Ganancia operacional / Total de activos (productividad de activos)
- **X4** = Patrimonio total / Total pasivos (solvencia)
- **X5** = Ingresos ordinarios / Total de activos (rotacion de activos)

**Clasificacion:**
| Zona | Rango de Z' | Etiqueta |
|---|---|---|
| Segura | Z' > 2.90 | Riesgo Bajo |
| Gris | 1.23 < Z' < 2.90 | Riesgo Medio |
| Peligro | Z' < 1.23 | Riesgo Alto |

**Ventaja**: Metodologia reconocida academicamente, reproducible.
**Limitacion**: Calibrada para empresas estadounidenses; puede requerir ajuste para el contexto colombiano.

#### Estrategia 2: Reglas Heuristicas Combinadas

Definir el riesgo basandose en umbrales multiples:

| Criterio | Riesgo Bajo | Riesgo Medio | Riesgo Alto |
|---|---|---|---|
| Razon corriente | > 1.5 | 1.0 - 1.5 | < 1.0 |
| Razon de deuda | < 0.5 | 0.5 - 0.7 | > 0.7 |
| Margen neto | > 5% | 0% - 5% | < 0% (perdida) |
| Cobertura de intereses | > 3.0 | 1.5 - 3.0 | < 1.5 |

Se asigna una puntuacion por criterio y se calcula el riesgo total.

**Ventaja**: Interpretable, adaptable al contexto local.
**Limitacion**: Subjetividad en los umbrales.

#### Estrategia 3: Clustering No Supervisado + Expertos

1. Aplicar K-Means o DBSCAN sobre los indicadores financieros
2. Analizar los clusters resultantes para identificar patrones de riesgo
3. Etiquetar los clusters con ayuda de criterio experto

**Ventaja**: No requiere umbrales predefinidos; descubre patrones naturales.
**Limitacion**: Requiere interpretacion posterior; los clusters pueden no alinearse con el concepto de "riesgo".

#### Recomendacion para la tesis

Usar una **combinacion de las estrategias 1 y 2**:
- Calcular el Z-Score de Altman como base principal
- Complementar con reglas heuristicas para validar la clasificacion
- Documentar ambos enfoques y comparar resultados

### 4.3 Features (Variables de Entrada para el Modelo)

Los **features** son las variables que el modelo usara para predecir el riesgo. Se organizan en tres niveles:

#### Nivel 1: Indicadores financieros directos (18 indicadores)

Los 18 indicadores calculados en la seccion 3 de este documento. Son las variables mas interpretables y directamente utiles.

#### Nivel 2: Variables derivadas (feature engineering)

| Feature | Calculo | Justificacion |
|---|---|---|
| **Variacion interanual** de cada indicador | `(indicador_t - indicador_t-1) / indicador_t-1` | Captura tendencias (mejora o deterioro) |
| **Tamano de la empresa** | `log(Total de activos)` | Normaliza la escala entre empresas de distintos tamanos |
| **Estructura de activos** | `Activos corrientes / Total de activos` | Indica que tan liquida es la estructura |
| **Estructura de deuda** | `Pasivos corrientes / Total pasivos` | Indica concentracion de deuda en corto plazo |
| **Margen EBITDA estimado** | `(Ganancia operacional + Depreciacion) / Ingresos` | Rentabilidad operativa sin efecto de amortizaciones |

#### Nivel 3: Variables categoricas

| Feature | Origen | Tratamiento |
|---|---|---|
| **Tipo de organizacion** | Camara de Comercio (ORGANIZACION) | One-hot encoding (SAS, Ltda, SA, etc.) |
| **Sector economico** | Camara de Comercio (CIIU) | One-hot encoding o target encoding |
| **Anio fiscal** | FECHA_CORTE | Variable numerica o one-hot |

### 4.4 Preprocesamiento de Datos

Antes de entrenar el modelo, se requieren los siguientes pasos:

#### 4.4.1 Manejo de valores faltantes

| Estrategia | Cuando usarla |
|---|---|
| **Eliminar columnas** con > 70% de nulos | Conceptos que no aplican a la mayoria de empresas (ej: "Activos biologicos") |
| **Imputar con 0** | Conceptos donde la ausencia implica "no tiene" (ej: inventarios para empresas de servicios) |
| **Imputar con mediana** del sector | Cuando se necesita conservar la observacion pero el dato falta |
| **Eliminar filas** | Solo si la empresa-anio tiene muy pocos datos para calcular indicadores basicos |

#### 4.4.2 Manejo de outliers

Los datos financieros suelen tener **outliers extremos** (empresas muy grandes o en crisis):

| Estrategia | Detalle |
|---|---|
| **Winsorizar** al percentil 1-99 | Reemplazar valores extremos por los percentiles |
| **Transformacion logaritmica** | Para variables con distribucion sesgada (activos, ingresos) |
| **Indicadores como ratios** | Ya normalizan el efecto de tamano (ROA, margenes) |

#### 4.4.3 Escalado de features

| Metodo | Cuando usarlo |
|---|---|
| **StandardScaler** (z-score) | Para modelos basados en distancia (SVM, KNN) o redes neuronales |
| **Sin escalado** | Para modelos basados en arboles (Random Forest, XGBoost) que son invariantes a escala |

### 4.5 Seleccion de Modelo

Para clasificacion de riesgo financiero, los modelos mas apropiados son:

| Modelo | Ventajas | Limitaciones | Recomendado |
|---|---|---|---|
| **Random Forest** | Robusto a outliers, maneja datos faltantes, interpretable via importancia de features | Puede sobre-ajustar con pocas observaciones | Si (baseline) |
| **XGBoost / LightGBM** | Alto rendimiento, maneja desbalance de clases, regularizacion incorporada | Mas hiperparametros que ajustar | Si (modelo principal) |
| **Regresion Logistica** | Muy interpretable (coeficientes como "odds ratios"), rapido de entrenar | Asume relacion lineal entre features y target | Si (benchmark simple) |
| **SVM** | Buen rendimiento en dimensiones altas | Menos interpretable, sensible a escalado | Opcional |
| **Red Neuronal** | Captura relaciones no lineales complejas | Requiere muchos datos; caja negra | No recomendado (dataset pequeno) |

#### Recomendacion para la tesis

1. **Regresion Logistica** como baseline (interpretable, sirve como referencia)
2. **Random Forest** como modelo intermedio (robusto, importancia de features)
3. **XGBoost** como modelo principal (mejor rendimiento esperado)
4. Comparar los 3 modelos con las mismas metricas

### 4.6 Evaluacion del Modelo

#### Metricas principales

| Metrica | Definicion | Por que es importante |
|---|---|---|
| **Accuracy** | Porcentaje de predicciones correctas | Vision general del rendimiento |
| **F1-Score (macro)** | Media armonica de precision y recall, promediada por clase | Importante si las clases estan desbalanceadas |
| **AUC-ROC** | Area bajo la curva ROC (one-vs-rest para multiclase) | Mide la capacidad discriminativa del modelo |
| **Matriz de Confusion** | Tabla de predicciones vs valores reales | Identifica que tipos de errores comete el modelo |

#### Validacion

| Estrategia | Detalle |
|---|---|
| **K-Fold Cross Validation** (k=5) | Divide los datos en 5 partes, entrena en 4 y valida en 1, rotando |
| **Stratified K-Fold** | Variante que mantiene la proporcion de clases en cada fold |
| **Leave-One-Group-Out** | Validar dejando fuera una empresa completa (evita data leakage temporal) |

> **Importante**: No usar train/test split simple por la naturaleza panel (empresa x anio) de los datos. Si una empresa aparece en train y test con distintos anios, el modelo puede "memorizar" patrones de la empresa en lugar de aprender patrones generales de riesgo.

### 4.7 Pipeline Propuesto

```
1. DATOS CRUDOS (dataset consolidado)
       |
2. CALCULO DE INDICADORES (18 indicadores financieros)
       |
3. FEATURE ENGINEERING (variaciones interanuales, log-tamano, categoricas)
       |
4. PREPROCESAMIENTO (imputacion, outliers, escalado)
       |
5. ETIQUETADO (Z-Score Altman + reglas heuristicas -> Bajo/Medio/Alto)
       |
6. ENTRENAMIENTO (Logistica, Random Forest, XGBoost)
       |
7. EVALUACION (F1-macro, AUC-ROC, Matriz de Confusion)
       |
8. MODELO FINAL (mejor modelo seleccionado)
       |
9. INTEGRACION WEB (API en Node.js que invoca el modelo Python)
```

### 4.8 Consideraciones Especificas para el Dataset

| Aspecto | Situacion | Mitigacion |
|---|---|---|
| **Tamano reducido** | ~300-450 observaciones empresa-anio (con datos corregidos) | Usar modelos simples, regularizacion fuerte, validacion cruzada |
| **Panel desbalanceado** | No todas las empresas tienen todos los anios | Validacion Leave-One-Group-Out por empresa |
| **Datos faltantes** | Completitud variable (20-73% segun estado financiero) | Seleccionar features con alta completitud; imputar con cuidado |
| **Clases desbalanceadas** | Probablemente pocas empresas en "riesgo alto" | SMOTE, class weights, o under-sampling de clase mayoritaria |
| **Estacionalidad economica** | COVID-19 afecto anios 2020-2021 | Incluir ANIO como feature o dummies para controlar efectos temporales |

---

## 5. Glosario de Terminos

| Termino | Definicion |
|---|---|
| **SIREM** | Sistema de Informacion y Reporte Empresarial (Superintendencia de Sociedades) |
| **NIIF** | Normas Internacionales de Informacion Financiera |
| **NIIF Pymes (Grupo 2)** | Normativa contable simplificada para pequenas y medianas empresas en Colombia |
| **NIT** | Numero de Identificacion Tributaria (identificador fiscal colombiano) |
| **DV** | Digito de verificacion del NIT |
| **CIIU** | Clasificacion Industrial Internacional Uniforme (codigo de actividad economica) |
| **Pivot** | Transformacion de datos del formato largo (vertical) al ancho (horizontal) |
| **Feature** | Variable de entrada para un modelo de Machine Learning |
| **Target** | Variable objetivo que el modelo intenta predecir |
| **ROA** | Return on Assets (retorno sobre activos) |
| **ROE** | Return on Equity (retorno sobre patrimonio) |
| **EBITDA** | Earnings Before Interest, Taxes, Depreciation and Amortization |
| **SMOTE** | Synthetic Minority Over-sampling Technique (genera datos sinteticos para clases minoritarias) |

---

*Documento creado: Febrero 2026*
*Proyecto de tesis - Universidad de Ibague*
*Datos fuente: SIREM (Supersociedades) + Camara de Comercio de Ibague*
