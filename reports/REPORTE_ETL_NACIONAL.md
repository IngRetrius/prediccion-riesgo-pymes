# Reporte de Resultados: ETL Nacional PYMES Colombia

**Notebook**: `etl_nacional_pymes.ipynb`
**Fecha de ejecucion**: 10 de febrero de 2026
**Datos fuente**: SIREM - Superintendencia de Sociedades (4 datasets, 9.17 GB)
**Periodo cubierto**: 2016 - 2024 (9 anos fiscales)

---

## 1. Resumen Ejecutivo

Se proceso exitosamente la totalidad de los datos financieros del SIREM para todas las empresas colombianas bajo **NIIF Pymes**, generando un dataset consolidado nacional que servira como base de entrenamiento para el modelo de Machine Learning.

| Metrica | Valor |
|---------|-------|
| **Observaciones empresa-ano** | 203,104 |
| **Empresas unicas** | 38,245 |
| **Columnas totales** | 230 (2 ID + 24 metadatos + 204 financieras) |
| **Periodo** | 2016 - 2024 (9 anos) |
| **Tamano consolidado** | 213.7 MB |
| **Tamano total archivos generados** | 367.1 MB |
| **Tiempo total de procesamiento** | ~92 s (carga) + ~11 s (pivoteo) |

### Comparacion con enfoque anterior (Ibague)

| Aspecto | Enfoque Ibague | Enfoque Nacional | Factor |
|---------|---------------|-----------------|--------|
| Empresas | 61 | 38,245 | **x627** |
| Observaciones empresa-ano | 359 | 203,104 | **x566** |
| Fuente de metadatos | Camara de Comercio Ibague | Caratula SIREM | Nacional |
| Conceptos financieros | ~85 | 204 (con sufijos) | Mas completo |

---

## 2. Datos Fuente Procesados

### 2.1 Archivos SIREM

| Dataset | Tamano | Filas totales | Filas NIIF Pymes | % Pymes | Encoding | Tiempo carga |
|---------|--------|--------------|-----------------|---------|----------|-------------|
| Caratula | 2.07 GB | 8,685,453 | 7,208,068 | 83.0% | UTF-8 | 21 s |
| Situacion Financiera | 4.10 GB | 17,851,220 | 14,725,475 | 82.5% | latin-1 | 43 s |
| Resultado Integral | 1.57 GB | 7,320,752 | 5,980,571 | 81.7% | latin-1 | 16 s |
| Flujo de Efectivo | 1.43 GB | 5,769,293 | 4,668,889 | 80.9% | latin-1 | 12 s |
| **TOTAL** | **9.17 GB** | **39,626,718** | **32,583,003** | **82.2%** | | **92 s** |

**Hallazgo**: Las NIIF Pymes representan consistentemente el ~82% de todos los registros del SIREM, confirmando que la mayoria de empresas que reportan a Supersociedades son PYMES.

### 2.2 Distribucion de PUNTO_ENTRADA

Los datos NIIF Pymes incluyen varios subtipos de reporte:

| Subtipo | % aprox. del total Pymes |
|---------|-------------------------|
| Individual Grupo 2 | 38-46% |
| Individuales | 27-35% |
| Separado Grupo 2 | 3.4-4.4% |
| Separados | 2.9-3.7% |
| Consolidado Grupo 2 | 1.4-1.5% |
| Consolidados | 0.9-1.3% |
| Combinado Grupo 2 | 0.1% |

> **Nota**: "Individual Grupo 2" y "Individuales" juntos representan ~73% de los datos Pymes. Los demas son reportes separados, consolidados y combinados.

---

## 3. Resultados por Dataset

### 3.1 Caratula (Metadatos)

**Proposito**: Fuente de metadatos de identificacion empresarial (reemplaza a la Camara de Comercio de Ibague para cobertura nacional).

| Metrica | Valor |
|---------|-------|
| Filas totales cargadas | 8,685,453 |
| Filas NIIF Pymes | 7,208,068 |
| Empresas unicas | 38,245 |
| Conceptos disponibles | 55 |
| Conceptos seleccionados | 24 |
| Resultado pivotado | 38,245 empresas x 25 columnas |
| Archivo generado | colombia_metadata_pymes.csv (15.5 MB) |

**Conceptos seleccionados** (por palabras clave): razon social, nombre, CIIU, actividad economica, departamento, municipio, direccion, tamano, tipo de sociedad, correo, telefono (24 de 55 conceptos).

**Metodo de pivoteo**: `aggfunc='last'` (valor mas reciente por empresa), ya que los metadatos no varian significativamente entre anos.

### 3.2 Estado de Situacion Financiera (Balance General)

**Proposito**: Posicion financiera — activos, pasivos, patrimonio.

| Metrica | Valor |
|---------|-------|
| Filas cargadas (Pymes) | 14,725,475 |
| Filas post-filtro PERIODO | 7,061,409 |
| Filas removidas (comparativos) | 7,664,066 (52.1%) |
| Empresas unicas | 38,233 |
| Combinaciones empresa-ano | 202,501 |
| Conceptos contables | 80 |
| Resultado pivotado | 202,501 filas x 82 columnas |
| Completitud | 37.7% |
| Duplicados NIT+ANIO+CONCEPTO | 883,611 de 6,115,199 (14.4%) |
| Tiempo de pivoteo | 5.3 s |
| Archivo generado | colombia_situacion_financiera_pymes.csv (62.4 MB) |

**Registros removidos por filtro de PERIODO**:
- "Periodo Anterior": 5,782,770 (comparativo del ano previo)
- "2015-dic-31": 690,227 (comparativo para reportes de 2016)
- "2015-ene-01": 642,103 (saldo de apertura comparativo)
- "2016-dic-31": 541,454 (comparativo en reportes de 2017)
- Otros: 7,512

**Conceptos clave verificados** (todos encontrados exitosamente):

| Concepto | Encontrado | Uso en indicadores |
|----------|-----------|-------------------|
| Activos corrientes totales | SI | Razon corriente, Prueba acida, Capital de trabajo |
| Total de activos | SI | ROA, Rotacion de activos |
| Pasivos corrientes totales | SI | Razon corriente, Capital de trabajo |
| Total pasivos | SI | Razon de deuda, Deuda/Patrimonio |
| Patrimonio total | SI | ROE, Apalancamiento |
| Inventarios corrientes | SI | Prueba acida, Rotacion de inventarios |
| Efectivo y equivalentes al efectivo | SI | Razon de efectivo |
| Cuentas comerciales por cobrar corrientes | SI | Rotacion de cartera |
| Ganancias acumuladas | SI | Z-Score Altman |
| Propiedades, planta y equipo | SI | Estructura de activos |

### 3.3 Estado de Resultado Integral (Perdidas y Ganancias)

**Proposito**: Desempeno financiero — ingresos, costos, utilidades.

| Metrica | Valor |
|---------|-------|
| Filas cargadas (Pymes) | 5,980,571 |
| Filas post-filtro PERIODO | 2,997,824 |
| Filas removidas (comparativos) | 2,982,747 (49.9%) |
| Empresas unicas | 38,245 |
| Combinaciones empresa-ano | 203,103 |
| Conceptos contables | 23 |
| Resultado pivotado | 203,103 filas x 25 columnas |
| Completitud | 55.4% |
| Duplicados NIT+ANIO+CONCEPTO | 383,778 de 2,585,991 (14.8%) |
| Tiempo de pivoteo | 2.1 s |
| Archivo generado | colombia_resultado_integral_pymes.csv (26.6 MB) |

**Registros removidos por filtro de PERIODO**:
- "Periodo Anterior": 2,493,192
- "2015": 267,028
- "2016": 221,038
- "2016-S2": 456
- "2015-S2": 277
- "2015-dic-31 / 2016-dic-31": 249

**Nota**: El Resultado Integral tiene la **mayor completitud** (55.4%) entre los tres estados financieros, lo cual es logico porque tiene menos conceptos (23 vs 80 vs 101) y la mayoria de empresas reportan los campos principales de ingresos y gastos.

**Conceptos clave** (15 de 23 marcados como clave):
- Ingresos de actividades ordinarias
- Costo de ventas
- Ganancia bruta
- Gastos de administracion / Gastos de ventas
- Ganancia (perdida) por actividades de operacion
- Costos financieros / Ingresos financieros
- Ganancia (perdida), antes de impuestos
- Ganancia (perdida) (utilidad neta)

### 3.4 Estado de Flujo de Efectivo

**Proposito**: Movimientos reales de dinero — operacion, inversion, financiacion.

| Metrica | Valor |
|---------|-------|
| Filas cargadas (Pymes) | 4,668,889 |
| Filas post-filtro PERIODO | 4,652,252 |
| Filas removidas | 16,637 (0.4%) |
| Empresas unicas | 38,245 |
| Combinaciones empresa-ano | 203,104 |
| Conceptos contables | 101 |
| Resultado pivotado | 203,104 filas x 103 columnas |
| Completitud | 18.9% |
| Duplicados NIT+ANIO+CONCEPTO | 688,032 de 3,869,087 (17.8%) |
| Tiempo de pivoteo | 3.6 s |
| Archivo generado | colombia_flujo_efectivo_pymes.csv (48.9 MB) |

**Anomalia documentada: NITs como PERIODO**

En el campo PERIODO del Flujo de Efectivo para 2016, en lugar de "Periodo Actual" o una fecha, aparecen **NITs de empresas** (numeros de 7-9 digitos). Esto afecta 543,879 filas (19,723 valores unicos de PERIODO).

- **Causa**: Error en el dataset fuente del SIREM para datos de 2016
- **FECHA_CORTE afectadas**: 2016 Dec 31, 2016 Jun 30, 2016 Sep 30, y otros meses de 2016
- **Impacto**: Ninguno en datos financieros — el campo VALOR es valido, solo PERIODO esta corrupto
- **Solucion aplicada**: El filtro inteligente conserva estos registros porque los NITs (ej: "800077198") no coinciden con el patron de fecha `^[0-9]{4}(-|$)` ni contienen "anterior"

**Nota sobre completitud**: El Flujo de Efectivo tiene la **menor completitud** (18.9%) porque tiene 101 conceptos y muchas PYMES solo reportan un subconjunto de las lineas del flujo de efectivo.

---

## 4. Consolidacion Final

### 4.1 Proceso de Merge

```
Situacion Financiera (202,501 x 82)  ──┐
                                        │
Resultado Integral (203,103 x 25)   ───┤── outer join ──► 203,104 x 206
                                        │
Flujo de Efectivo (203,104 x 103)   ───┘
                                              │
                                              ├── left join ──► 203,104 x 230
                                              │
Metadata (38,245 x 25)  ──────────────────────┘
```

**Conceptos compartidos entre estados**: Solo 1 concepto aparece en mas de un estado financiero con nombre identico:
- `Ganancia (perdida)` — aparece en Resultado Integral y Flujo de Efectivo → se crea sufijo `_flujo`

### 4.2 Dataset Consolidado

| Metrica | Valor |
|---------|-------|
| Filas | 203,104 |
| Columnas | 230 |
| Columnas de identificacion | 2 (NIT_LIMPIO, ANIO) |
| Columnas de metadatos | 24 |
| Columnas financieras | 204 |
| Empresas con metadatos | 173,403 de 203,104 filas (85.4%) |

### 4.3 Cobertura Temporal

**Empresas por ano**:

| Ano | Empresas | Tendencia |
|-----|----------|-----------|
| 2016 | 19,703 | Base |
| 2017 | 16,419 | -16.7% |
| 2018 | 15,947 | -2.9% |
| 2019 | 21,859 | +37.1% |
| 2020 | 25,994 | +18.9% |
| 2021 | 25,214 | -3.0% |
| 2022 | 26,925 | +6.8% |
| 2023 | 26,966 | +0.2% (pico) |
| 2024 | 24,077 | -10.7% |

**Observaciones**:
- Caida en 2017-2018 (posiblemente por transicion NIIF o ajustes en reporte)
- Salto significativo en 2019 (+37%) — puede reflejar mayor adopcion de NIIF Pymes
- Estabilizacion en 2020-2023 (~25,000-27,000 empresas)
- 2024 muestra ligera caida, posiblemente por reportes aun no radicados al momento de descarga

**Profundidad temporal por empresa**:

| Anos de datos | Empresas | % del total | Acumulado |
|--------------|----------|-------------|-----------|
| 9 (completo) | 9,457 | 24.7% | 24.7% |
| 8 | 1,713 | 4.5% | 29.2% |
| 7 | 1,716 | 4.5% | 33.7% |
| 6 | 6,624 | 17.3% | 51.0% |
| 5 | 4,104 | 10.7% | 61.8% |
| 4 | 2,173 | 5.7% | 67.4% |
| 3 | 3,301 | 8.6% | 76.1% |
| 2 | 4,259 | 11.1% | 87.2% |
| 1 | 4,898 | 12.8% | 100.0% |

- **Promedio**: 5.3 anos por empresa
- **Mediana**: 6 anos
- **24.7%** de las empresas tienen los 9 anos completos (2016-2024) — ideal para analisis de tendencias
- **51%** de empresas tienen 6+ anos de datos — buena profundidad para ML

---

## 5. Archivos Generados

| # | Archivo | Contenido | Filas | Columnas | Tamano |
|---|---------|-----------|-------|----------|--------|
| 1 | `colombia_metadata_pymes.csv` | Metadatos de empresas | 38,245 | 25 | 15.5 MB |
| 2 | `colombia_situacion_financiera_pymes.csv` | Balance general pivotado | 202,501 | 82 | 62.4 MB |
| 3 | `colombia_resultado_integral_pymes.csv` | Perdidas y ganancias pivotado | 203,103 | 25 | 26.6 MB |
| 4 | `colombia_flujo_efectivo_pymes.csv` | Flujo de efectivo pivotado | 203,104 | 103 | 48.9 MB |
| 5 | `colombia_consolidado_pymes.csv` | Todo unificado | 203,104 | 230 | 213.7 MB |
| | **TOTAL** | | | | **367.1 MB** |

Todos los archivos se encuentran en: `C:\Users\USUARIO1\Documents\Tesis\data\`

---

## 6. Ventajas del Enfoque Nacional

### 6.1 Para el Modelo de Machine Learning

1. **Volumen de datos suficiente**: 203,104 observaciones empresa-ano (vs 359 anteriores). Permite entrenamiento robusto de modelos como Random Forest, XGBoost, e incluso redes neuronales.

2. **Representatividad estadistica**: 38,245 empresas cubren multiples sectores economicos (CIIU), tamanos y regiones. Los patrones aprendidos seran generalizables.

3. **Datos de panel**: El 51% de empresas tiene 6+ anos de datos, lo que permite:
   - Calcular variaciones interanuales
   - Detectar tendencias de deterioro financiero
   - Construir features temporales (promedios moviles, ratios de crecimiento)

4. **Validacion cruzada real**: Se puede usar hold-out por ano (ej: entrenar con 2016-2022, validar con 2023-2024) en lugar de split aleatorio, lo cual es mas realista para prediccion financiera.

### 6.2 Para el Analisis Financiero

1. **Benchmarks sectoriales nacionales**: Con miles de empresas por sector, los promedios sectoriales seran estadisticamente significativos (vs 61 empresas donde un outlier distorsiona todo).

2. **Ibague como caso de validacion**: Las 61 empresas de Ibague se pueden comparar contra los promedios nacionales por sector, generando un analisis mucho mas rico.

3. **Deteccion de outliers robusta**: Con 38,245 empresas, se pueden establecer rangos intercuartilicos confiables para cada indicador por sector.

### 6.3 Para la Tesis

1. **Mayor rigor academico**: Trabajar con todo el universo de empresas NIIF Pymes elimina el sesgo de seleccion.
2. **Metodologia reproducible**: El mismo ETL se puede aplicar cuando Supersociedades publique datos de 2025+.
3. **Contribucion original**: Un dataset consolidado de 203K observaciones financieras de PYMES colombianas es un activo de investigacion significativo.

---

## 7. Puntos Criticos y Riesgos

### 7.1 Completitud de Datos

| Estado Financiero | Completitud | Evaluacion |
|-------------------|-------------|------------|
| Resultado Integral | 55.4% | Aceptable — conceptos principales bien cubiertos |
| Situacion Financiera | 37.7% | Moderado — muchos conceptos detallados vacios |
| Flujo de Efectivo | 18.9% | **Bajo** — mayoria de conceptos no reportados |

**Implicacion**: No todos los 204 conceptos financieros son utiles. Para los 18 indicadores financieros planeados, se deben verificar que los conceptos especificos necesarios tengan cobertura suficiente. Los conceptos del Flujo de Efectivo probablemente requieran imputacion o exclusion selectiva.

**Recomendacion**: Antes de calcular indicadores, verificar la cobertura por concepto clave (ej: que % de las 203K filas tiene "Activos corrientes totales" no nulo).

### 7.2 Duplicados en el Pivoteo

| Dataset | Duplicados | % |
|---------|-----------|---|
| Situacion Financiera | 883,611 | 14.4% |
| Resultado Integral | 383,778 | 14.8% |
| Flujo de Efectivo | 688,032 | 17.8% |

Se resolvieron con `aggfunc='first'` (primer valor encontrado). Esto puede introducir inconsistencias si los duplicados tienen valores distintos. Las causas posibles de duplicacion incluyen:

- Empresas que reportan correciones o enmiendas
- Multiples radicados para la misma empresa en el mismo ano
- Subtipos de reporte (Individual Grupo 2, Individuales, Separados, etc.) que generan registros paralelos para la misma empresa

**Recomendacion**: Investigar si los duplicados provienen de distintos subtipos de PUNTO_ENTRADA. Si es asi, se podria priorizar un subtipo especifico (ej: "Individual Grupo 2" sobre "Individuales") en lugar de tomar el primero arbitrariamente.

### 7.3 Empresas sin Metadatos

- 203,104 filas financieras totales
- 173,403 filas con metadatos (85.4%)
- **29,701 filas sin metadatos** (14.6%)

Estas filas corresponden a empresas que aparecen en los estados financieros pero no en la Caratula (o cuyo NIT no coincide). Para el modelo ML, los indicadores financieros seguiran siendo calculables, pero no se podra clasificar por sector CIIU, departamento o tamano.

**Recomendacion**: Identificar si las 29,701 filas sin metadatos corresponden a un ano o grupo especifico, y si es viable recuperar los metadatos de otra fuente.

### 7.4 Diferencias de Filas entre Estados

| Dataset | Combinaciones empresa-ano |
|---------|--------------------------|
| Situacion Financiera | 202,501 |
| Resultado Integral | 203,103 |
| Flujo de Efectivo | 203,104 |
| **Consolidado (outer)** | **203,104** |

La Situacion Financiera tiene 603 combinaciones empresa-ano menos que los otros dos. Esto significa que hay 603 observaciones donde existe Resultado Integral o Flujo de Efectivo pero **no** el balance general. Estos registros tendran NaN en las 80 columnas del balance.

**Recomendacion**: Para el calculo de indicadores como razon corriente, ROA, o Z-Score Altman (que requieren datos del balance), estas 603 filas incompletas deberan excluirse o tratarse con imputacion.

### 7.5 Anomalia de PERIODO en Flujo de Efectivo

- 543,879 filas con NITs como valor de PERIODO (solo 2016)
- 19,723 valores unicos anomalos

Aunque los datos financieros son validos, esta anomalia indica problemas de calidad en la fuente SIREM para 2016. Es posible que existan otros problemas de calidad no detectados.

### 7.6 TAXONOMIA Truncada

- 7,834,219 filas en Situacion Financiera tienen TAXONOMIA terminada en "Corte 202" (sin ano completo)
- Esto corresponde al 53.2% de los datos Pymes del balance
- **Solucion aplicada**: Uso de FECHA_CORTE como fuente del ano fiscal (contiene el ano completo para el 100% de registros)

**Impacto**: Si algun analisis futuro necesita la TAXONOMIA exacta (ej: para identificar la version del marco contable), los datos de 2021+ no seran distinguibles.

### 7.7 FECHA_CORTE con Multiples Fechas por Ano

- FECHA_CORTE tiene 71 valores unicos (no solo 9 cortes al 31 de diciembre)
- Incluye fechas intermedias (junio, septiembre, etc.)
- Estas corresponden a empresas con cierres fiscales atipicos o reportes parciales

**Impacto**: El ano fue extraido correctamente de FECHA_CORTE, pero para empresas con cierre en junio (ej: "2016 Jun 30"), los datos financieros cubren un periodo diferente al ano calendario. Esto puede generar comparaciones desalineadas entre empresas.

---

## 8. Calidad de Datos — Resumen

| Dimension | Estado | Detalle |
|-----------|--------|---------|
| **Cobertura temporal** | Buena | 9 anos (2016-2024), promedio 5.3 anos/empresa |
| **Cobertura de empresas** | Excelente | 38,245 empresas NIIF Pymes (universo completo) |
| **Integridad de NIT** | Excelente | 100% de NITs son de 9 digitos tras limpieza |
| **Encoding** | Resuelto | UTF-8 para Caratula, latin-1 para estados financieros |
| **Duplicados** | Gestionado | 14-18% de duplicados resueltos con aggfunc='first' |
| **Completitud de celdas** | Variable | 18.9% - 55.4% segun estado financiero |
| **Metadatos** | Buena | 85.4% de observaciones con metadatos |
| **Anomalias conocidas** | Documentadas | PERIODO corrupto en Flujo 2016, TAXONOMIA truncada 2021+ |

---

## 9. Proximos Pasos

### Inmediatos (siguiente notebook)

1. **Validacion con Ibague**: Cruzar el dataset nacional con la Camara de Comercio de Ibague para identificar las 61 empresas ibaguenas dentro de las 38,245 nacionales.

2. **Analisis de completitud por concepto clave**: Antes de calcular indicadores, verificar que los 18 conceptos necesarios tengan cobertura suficiente (>80% de filas no nulas).

3. **Investigar duplicados**: Determinar si los duplicados del pivoteo provienen de multiples subtipos de PUNTO_ENTRADA y definir una politica de priorizacion.

### Calculo de Indicadores Financieros

4. **18 indicadores** en 4 categorias:
   - **Liquidez** (4): Razon corriente, prueba acida, capital de trabajo, razon de efectivo
   - **Solvencia** (4): Endeudamiento total, apalancamiento, cobertura de intereses, deuda/patrimonio
   - **Rentabilidad** (5): Margen bruto, margen operacional, margen neto, ROA, ROE
   - **Actividad** (5): Rotacion de cartera, inventarios, activos, proveedores, ciclo operativo

### Etiquetado y Modelo ML

5. **Etiquetado de riesgo**: Z-Score de Altman adaptado + reglas heuristicas → clasificacion Bajo/Medio/Alto riesgo.

6. **Entrenamiento de modelos**: Regresion Logistica, Random Forest, XGBoost sobre las 203K observaciones. Validacion temporal (entrenar 2016-2022, probar 2023-2024).

### Sistema Web

7. **Arquitectura web**: React (frontend) + Node.js (backend) + PostgreSQL (base de datos). Integrar modelo ML entrenado para predicciones en tiempo real.

---

## 10. Detalles Tecnicos del ETL

### 10.1 Funciones Auxiliares Implementadas

| Funcion | Descripcion | Uso |
|---------|-------------|-----|
| `limpiar_nit(df)` | Elimina comas del NIT, deja 9 digitos | Todos los datasets |
| `extraer_anio(df)` | Extrae ano de FECHA_CORTE con regex `[0-9]{4}` | Todos los datasets |
| `filtrar_periodo(df)` | Remueve "Periodo Anterior" y comparativos | 3 estados financieros |
| `analizar_cobertura(df)` | Muestra empresas/ano y profundidad temporal | 3 estados financieros |
| `pivotar_estado(df)` | Transforma vertical → horizontal con `pivot_table` | 3 estados financieros |

### 10.2 Estrategia de Memoria

- Procesamiento secuencial: un dataset a la vez (no se cargan los 4 simultaneamente)
- Liberacion con `del df` + `gc.collect()` despues de pivotar cada estado
- Solo se conservan en memoria: los 3 pivotados + metadatos
- Pico de memoria estimado: ~6-8 GB (al cargar Situacion Financiera de 4.10 GB)

### 10.3 Filtro Inteligente de PERIODO

```
Conserva:
  - "Periodo Actual" (datos corrientes 2018+)
  - Fechas que coinciden con ANIO (ej: "2016-dic-31" cuando ANIO=2016)
  - NITs como PERIODO (datos validos con campo corrupto)

Remueve:
  - "Periodo Anterior" (comparativo explicito)
  - Fechas/anos que NO coinciden con ANIO (comparativos implicitos)
```

### 10.4 Estrategia de Consolidacion

- **Merge de estados**: `outer join` sobre `[NIT_LIMPIO, ANIO]` — conserva todas las combinaciones
- **Merge de metadatos**: `left join` sobre `NIT_LIMPIO` — solo agrega si existe metadata
- **Conflicto de nombres**: Sufijos `_resultado` y `_flujo` para conceptos duplicados entre estados
- **Orden de columnas**: Identificacion → Metadatos → Financieros
