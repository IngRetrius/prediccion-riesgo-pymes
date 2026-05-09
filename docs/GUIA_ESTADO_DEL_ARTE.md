# Guia Completa para Elaborar un Estado del Arte con Calidad de Publicacion Academica

**Aplicacion**: Tesis de grado y articulos cientificos
**Contexto de referencia**: "Diseno e Implementacion de un Sistema Web de Analisis Financiero con Modelo de Machine Learning para la Evaluacion de Riesgo en PYMES del Municipio de Ibague, Tolima"

---

## 1. Que es el Estado del Arte y Por Que es Critico

### 1.1 Definicion

El estado del arte es una revision sistematica, critica y sintetica de la literatura cientifica existente sobre un tema de investigacion. No es un resumen de lecturas: es un **mapa argumentativo** que demuestra que el investigador comprende el campo, identifica lo que ya se ha hecho, lo que falta por hacer y donde se posiciona su trabajo.

### 1.2 Proposito en una tesis de grado

| Funcion | Que logra |
|---------|-----------|
| **Contextualizar** | Ubica el problema dentro de un campo disciplinar con antecedentes verificables |
| **Justificar** | Demuestra que existe un vacio real que la tesis viene a llenar |
| **Fundamentar decisiones** | Explica por que se eligio cierta metodologia, tecnologia o enfoque sobre otros |
| **Demostrar dominio** | Evidencia que el tesista conoce la discusion academica vigente |
| **Evitar duplicacion** | Confirma que el trabajo no repite algo ya publicado |

### 1.3 Proposito en un articulo cientifico

En articulos para revistas indexadas (Scopus, WoS), el estado del arte cumple funciones adicionales:

- **Posicionar la contribucion**: Los revisores (peer review) evaluaran si el autor conoce los trabajos mas relevantes y recientes del area.
- **Establecer la linea base**: Define contra que resultados previos se comparara el trabajo propuesto.
- **Demostrar novedad**: Si el estado del arte no evidencia un vacio claro, el articulo sera rechazado por falta de originalidad.

> **Regla practica**: Un estado del arte de tesis suele tener 15-30 paginas. En un articulo cientifico, se condensa a 2-4 paginas (seccion "Related Work" o "Literature Review"), pero con la misma rigurosidad.

---

## 2. Metodologia de Busqueda de Literatura

### 2.1 Bases de datos academicas recomendadas

| Base de datos | Tipo | Fortaleza | Acceso |
|---------------|------|-----------|--------|
| **Scopus** | Multidisciplinar | Mayor cobertura de revistas indexadas, metricas de citacion | Institucional |
| **Web of Science (WoS)** | Multidisciplinar | Indices de impacto JCR, alta selectividad | Institucional |
| **IEEE Xplore** | Ingenieria y computacion | Conferencias y revistas IEEE, estandar en sistemas y ML | Institucional/parcial libre |
| **Google Scholar** | Agregador | Cobertura amplia, incluye tesis y preprints | Libre |
| **ScienceDirect** | Multidisciplinar (Elsevier) | Revistas de finanzas, economia, ingenieria | Institucional |
| **SpringerLink** | Multidisciplinar | Libros, conferencias LNCS | Institucional |
| **arXiv** | Preprints | ML y computacion de vanguardia, sin peer review | Libre |
| **SSRN** | Economia y finanzas | Working papers de finanzas y riesgo | Libre |
| **Redalyc / SciELO** | Iberoamerica | Investigacion latinoamericana, acceso abierto | Libre |
| **Dialnet** | Iberoamerica | Tesis doctorales y revistas en espanol | Libre |

### 2.2 Estrategia de busqueda por fases

```
FASE 1: Busqueda exploratoria (Google Scholar)
   Objetivo: Mapear el campo, identificar autores clave, terminos tecnicos
   Duracion: 2-3 dias
   Resultado: Lista preliminar de 40-60 documentos

FASE 2: Busqueda sistematica (Scopus + WoS + IEEE)
   Objetivo: Recuperar literatura rigurosa con criterios reproducibles
   Duracion: 3-5 dias
   Resultado: Corpus refinado de 60-100 documentos

FASE 3: Busqueda por referencias cruzadas (snowballing)
   Objetivo: Encontrar trabajos citados por los articulos clave (backward)
             y trabajos que citan a los articulos clave (forward)
   Duracion: 2-3 dias
   Resultado: 10-20 documentos adicionales de alta relevancia

FASE 4: Busqueda de contexto local (Redalyc, SciELO, repositorios)
   Objetivo: Encontrar investigaciones sobre PYMES colombianas, NIIF, riesgo
   Duracion: 1-2 dias
   Resultado: 5-15 documentos de contexto latinoamericano/colombiano
```

### 2.3 Palabras clave y cadenas de busqueda

#### Idioma ingles (bases internacionales)

**Eje 1 — Machine Learning + Riesgo financiero:**
```
("machine learning" OR "deep learning" OR "random forest" OR "XGBoost"
 OR "gradient boosting" OR "ensemble methods")
AND
("financial risk" OR "credit risk" OR "bankruptcy prediction"
 OR "financial distress" OR "default prediction" OR "risk assessment")
```

**Eje 2 — PYMES / SMEs:**
```
("SME" OR "SMEs" OR "small and medium enterprises"
 OR "small business" OR "micro-enterprises")
AND
("financial analysis" OR "financial health" OR "financial indicators"
 OR "financial ratios")
```

**Eje 3 — Sistemas web + analisis financiero:**
```
("web application" OR "web system" OR "web-based" OR "dashboard")
AND
("financial analysis" OR "financial assessment" OR "risk evaluation")
```

**Eje 4 — Z-Score y modelos clasicos:**
```
("Altman Z-Score" OR "Ohlson O-Score" OR "Zmijewski"
 OR "bankruptcy model" OR "scoring model")
AND
("SME" OR "emerging market" OR "developing country" OR "Latin America")
```

#### Idioma espanol (bases iberoamericanas)

```
"analisis financiero" AND ("PYMES" OR "pequenas empresas" OR "microempresas")
"riesgo financiero" AND ("machine learning" OR "aprendizaje automatico")
"indicadores financieros" AND ("clasificacion" OR "prediccion" OR "riesgo")
"NIIF" AND ("PYMES" OR "estados financieros" OR "Colombia")
"sistema web" AND ("analisis financiero" OR "evaluacion de riesgo")
```

### 2.4 Criterios de inclusion y exclusion

| Criterio | Inclusion | Exclusion |
|----------|-----------|-----------|
| **Periodo** | 2015-2026 (priorizar 2019-2026) | Anteriores a 2015 (salvo seminales como Altman 1968) |
| **Tipo de documento** | Articulos en revista indexada, conferencias IEEE/ACM, tesis doctorales | Blogs, notas de prensa, white papers comerciales, Wikipedia |
| **Idioma** | Ingles, espanol | Otros idiomas (salvo traduccion disponible) |
| **Peer review** | Si (Scopus/WoS/IEEE) | Preprints sin validar (arXiv solo como complemento) |
| **Relevancia tematica** | ML aplicado a finanzas, riesgo de PYMES, sistemas web financieros | ML generico sin aplicacion financiera, finanzas sin componente analitico/tecnologico |
| **Contexto geografico** | Cualquier pais (priorizar emergentes y Latinoamerica) | Ninguna exclusion geografica |

### 2.5 Gestion de referencias

**Herramientas recomendadas:**

| Herramienta | Ventaja | Costo |
|-------------|---------|-------|
| **Zotero** | Gratuito, plugin de navegador, genera bibliografias APA/IEEE automaticamente | Gratis |
| **Mendeley** | Integrado con Scopus/Elsevier, anotacion de PDFs | Gratis (basico) |
| **EndNote** | Estandar en muchas universidades | Licencia institucional |

**Flujo recomendado:**
1. Guardar cada articulo en Zotero/Mendeley con metadatos completos
2. Etiquetar por categoria tematica (ej: "ML-riesgo", "PYMES-indicadores", "sistemas-web")
3. Anotar hallazgos clave directamente en el PDF
4. Exportar bibliografia en formato APA 7 o IEEE segun la revista destino

---

## 3. Estrategias para Organizar la Informacion

### 3.1 Enfoque tematico (recomendado para esta tesis)

Agrupa la literatura por **ejes tematicos** que reflejan los componentes del trabajo. Este enfoque permite mostrar como convergen distintas disciplinas en tu investigacion.

**Estructura sugerida para esta tesis:**

```
Estado del Arte
|
|-- 3.1 Analisis financiero y evaluacion de riesgo en PYMES
|   |-- Indicadores financieros tradicionales (liquidez, solvencia, rentabilidad)
|   |-- Modelos clasicos de prediccion de quiebra (Altman, Ohlson, Zmijewski)
|   |-- Contexto PYMES: limitaciones de datos, informalidad, NIIF
|
|-- 3.2 Machine Learning aplicado a la prediccion de riesgo financiero
|   |-- Modelos supervisados: Regresion Logistica, SVM, Random Forest, XGBoost
|   |-- Comparativas de rendimiento entre modelos
|   |-- Feature engineering con indicadores financieros
|   |-- Manejo de desbalance de clases en datos financieros
|
|-- 3.3 Sistemas web para visualizacion y analisis financiero
|   |-- Dashboards financieros y herramientas de decision
|   |-- Arquitecturas web para integracion de modelos ML
|   |-- Experiencia de usuario en herramientas financieras para PYMES
|
|-- 3.4 Contexto colombiano y latinoamericano
|   |-- PYMES en Colombia: estructura economica, mortalidad empresarial
|   |-- Marco normativo NIIF en Colombia
|   |-- Estudios previos sobre riesgo financiero en PYMES colombianas
```

### 3.2 Enfoque cronologico (complementario)

Util para mostrar la **evolucion** de un tema. Se puede usar dentro de una seccion tematica.

**Ejemplo — Evolucion de modelos de prediccion de quiebra:**

```
1968      Altman Z-Score (Analisis discriminante multivariado)
1980      Ohlson O-Score (Regresion logistica)
1984      Zmijewski (Probit)
2005-2010 Primeras aplicaciones de SVM y redes neuronales
2010-2015 Random Forest supera a modelos clasicos en varios estudios
2015-2020 XGBoost y LightGBM dominan competencias (Kaggle, KDD)
2020-2026 Modelos hibridos, deep learning tabular, interpretabilidad (SHAP)
```

### 3.3 Enfoque metodologico (complementario)

Agrupa los trabajos por **metodo empleado**, permitiendo comparar resultados entre enfoques.

**Ejemplo — Tabla comparativa de modelos:**

| Autor(es) | Ano | Pais | Datos | Modelo(s) | Mejor AUC | Variable objetivo |
|-----------|-----|------|-------|-----------|-----------|-------------------|
| Wang et al. | 2020 | China | 12,000 PYMES | XGBoost, RF, LR | 0.89 (XGB) | Default a 1 ano |
| Barboza et al. | 2017 | EE.UU. | 10,000 empresas | SVM, Bagging, RF, Boosting | 0.94 (Bagging) | Bancarrota |
| Perez & Garcia | 2021 | Colombia | 500 PYMES | LR, RF | 0.82 (RF) | Riesgo PYME |
| Propuesta (tesis) | 2026 | Colombia | 38,245 PYMES | LR, RF, XGBoost | Por determinar | Riesgo financiero |

> Esta tabla se actualiza a medida que se leen mas articulos. En la version final del estado del arte, se convierte en un activo visual poderoso que demuestra dominio del campo.

### 3.4 Matriz de mapeo de literatura

Herramienta de trabajo para organizar lecturas antes de redactar:

| Referencia | Eje tematico | Aporte principal | Metodologia | Limitacion identificada | Relacion con mi tesis |
|------------|-------------|-------------------|-------------|------------------------|----------------------|
| Altman (1968) | Riesgo | Z-Score original | Discriminante | Solo grandes empresas, datos de EE.UU. | Base teorica del etiquetado |
| Wang et al. (2020) | ML+Riesgo | XGBoost superior en PYMES chinas | XGBoost, RF, LR | Datos de un solo pais, sin sistema web | Similar a mi enfoque pero sin sistema |
| ... | ... | ... | ... | ... | ... |

---

## 4. Analisis Critico vs. Resumen Descriptivo

### 4.1 Lo que NO es un estado del arte

**Resumen descriptivo** (mal ejemplo):

> "Lopez (2019) estudio el riesgo financiero en 200 PYMES de Mexico usando regresion logistica y obtuvo una precision del 78%. Por otro lado, Martinez (2020) analizo 150 PYMES de Chile usando Random Forest y obtuvo una precision del 83%. Adicionalmente, Rodriguez (2021) aplico XGBoost a 300 PYMES de Colombia y reporto un AUC de 0.87."

**Problema**: Es una lista de resumenes independientes. No hay dialogo entre autores, no hay posicion del investigador, no hay sintesis.

### 4.2 Lo que SI es un estado del arte

**Analisis critico** (buen ejemplo):

> "La prediccion de riesgo financiero en PYMES ha sido abordada con creciente sofisticacion metodologica. Los modelos clasicos de regresion logistica (Lopez, 2019; Chen & Li, 2018) ofrecen interpretabilidad pero alcanzan precisiones moderadas (75-80%) cuando se aplican a datos de PYMES, donde la variabilidad financiera es mayor que en grandes corporaciones. Los metodos de ensamble —particularmente Random Forest (Martinez, 2020) y XGBoost (Rodriguez, 2021; Wang et al., 2020)— han demostrado mejoras consistentes de 5-10 puntos porcentuales en AUC, aunque su rendimiento depende criticamente de la calidad del feature engineering y del tamano muestral. Un patron recurrente en la literatura es la limitacion del tamano de los datasets: la mayoria de estudios trabajan con 150-500 empresas (Lopez, 2019; Martinez, 2020), lo cual restringe la capacidad de generalizacion. El presente trabajo aborda esta brecha con un dataset de 38,245 PYMES colombianas y 203,104 observaciones empresa-ano, un orden de magnitud superior a los estudios previos en Latinoamerica."

**Por que funciona**:
- Agrupa autores por hallazgo comun (no por articulo individual)
- Identifica un patron ("mejoras consistentes de 5-10 puntos")
- Senala una limitacion recurrente (tamano de datasets)
- Posiciona la tesis como respuesta a esa limitacion

### 4.3 Plantillas de parrafos para analisis critico

**Plantilla 1 — Comparar enfoques:**

> "Dos corrientes principales han abordado [tema]. Por un lado, [Autor1 (ano)] y [Autor2 (ano)] proponen [enfoque A], argumentando que [justificacion]. Por otro lado, [Autor3 (ano)] y [Autor4 (ano)] optan por [enfoque B], que ofrece [ventaja]. Sin embargo, ambas corrientes comparten la limitacion de [limitacion], lo cual sugiere que [oportunidad de investigacion]."

**Plantilla 2 — Identificar evolucion:**

> "La investigacion en [tema] ha transitado de [enfoque inicial] (Autor1, ano) hacia [enfoque actual] (Autor2, ano; Autor3, ano). Este cambio responde a [razon: disponibilidad de datos / avances computacionales / nuevas regulaciones]. No obstante, la aplicacion de [enfoque actual] en el contexto de [tu contexto especifico] permanece poco explorada, con apenas [N] estudios publicados en [region/sector]."

**Plantilla 3 — Senalar vacio:**

> "A pesar de los avances en [tema], se identifican tres vacios significativos en la literatura: (1) la mayoria de estudios se concentran en [paises/contextos], con escasa evidencia para [tu contexto]; (2) los tamanos muestrales suelen ser inferiores a [N], lo que limita la [generalizacion/robustez]; y (3) pocos trabajos integran [componente A] con [componente B] en una solucion unificada, como se propone en el presente estudio."

**Plantilla 4 — Sintetizar resultados cuantitativos:**

> "Los resultados reportados en la literatura muestran que [modelo X] alcanza un rendimiento promedio de [metrica] en tareas de [tarea], con valores que oscilan entre [rango] dependiendo del tamano del dataset y las variables predictoras utilizadas (Tabla N). [Autor1 (ano)] reporta el mejor desempeno con [valor], aunque sobre un dataset de [N] observaciones de [pais], lo cual dificulta la comparacion directa con contextos de [tu contexto]."

**Plantilla 5 — Posicionar tu trabajo:**

> "El presente trabajo se diferencia de los antecedentes en tres aspectos: (1) [diferencia 1 — ej: escala de datos]; (2) [diferencia 2 — ej: integracion en sistema web]; y (3) [diferencia 3 — ej: contexto geografico]. Mientras que [Autor (ano)] y [Autor (ano)] abordan [problema] de forma aislada, este estudio propone una solucion integral que combina [componentes]."

---

## 5. Identificar Vacios, Tendencias y Debates

### 5.1 Como identificar vacios de investigacion

**Metodo 1 — Buscar lo que "no se ha hecho":**

Mientras lees cada articulo, hazte estas preguntas:
- ¿Se ha aplicado este metodo en PYMES? ¿En Latinoamerica? ¿En Colombia?
- ¿El tamano del dataset es representativo o es una muestra pequena?
- ¿Los resultados se integran en una herramienta usable o quedan en teoria?
- ¿Se usa informacion financiera bajo NIIF o bajo normas locales anteriores?
- ¿Se considera la dimension temporal (datos de panel) o solo un corte transversal?

**Metodo 2 — Leer las secciones "Future Work" y "Limitations":**

Los propios autores senalan lo que queda pendiente. Documenta sistematicamente estas declaraciones:

| Articulo | Limitacion declarada | Trabajo futuro sugerido | ¿Mi tesis lo aborda? |
|----------|---------------------|------------------------|---------------------|
| Wang (2020) | "Solo datos chinos" | Replicar en otros paises | Si — Colombia |
| Lopez (2019) | "Muestra de 200 empresas" | Usar datasets mas grandes | Si — 38,245 empresas |
| Martinez (2020) | "Sin interfaz de usuario" | Desarrollar dashboard | Si — sistema web |

**Metodo 3 — Buscar intersecciones vacias:**

Dibuja una matriz de los ejes tematicos de tu tesis y marca que combinaciones tienen cobertura en la literatura:

| | PYMES generico | PYMES Colombia | PYMES + NIIF |
|---|:-:|:-:|:-:|
| **Indicadores financieros** | Abundante | Moderado | Escaso |
| **ML + riesgo** | Abundante | Escaso | Muy escaso |
| **Sistema web** | Moderado | Escaso | Inexistente |
| **ML + sistema web + PYMES** | Escaso | Muy escaso | **Vacio = tu tesis** |

### 5.2 Tendencias actuales (2020-2026) a documentar

Para el tema de esta tesis, las tendencias clave que debes rastrear son:

1. **Interpretabilidad de ML (XAI)**: SHAP, LIME para explicar predicciones de riesgo. Los reguladores financieros exigen que los modelos sean explicables.

2. **Modelos de ensamble como estandar**: XGBoost, LightGBM y CatBoost dominan las tareas de clasificacion tabular, superando a deep learning en datos estructurados.

3. **Open finance y datos abiertos**: Mas paises publican datos financieros (como el SIREM en Colombia), habilitando investigacion a gran escala.

4. **AutoML**: Herramientas que automatizan la seleccion de modelos y optimizacion de hiperparametros.

5. **Datos de panel en ML**: Incorporar la dimension temporal (multiples anos por empresa) en lugar de cortes transversales.

6. **MLOps y despliegue**: Transicion de modelos en Jupyter notebooks a servicios web productivos.

### 5.3 Debates activos en el campo

| Debate | Posicion A | Posicion B | Tu posicion sugerida |
|--------|-----------|-----------|---------------------|
| ¿ML supera siempre a modelos clasicos? | Si, en datasets grandes (Barboza 2017, Wang 2020) | No siempre, depende de calidad de features (Hand 2006) | Comparar ambos empiricamente |
| ¿Precision vs. interpretabilidad? | Priorizar precision (modelo caja negra) | Priorizar interpretabilidad (regulacion financiera) | Usar SHAP para hacer interpretable un modelo preciso |
| ¿Z-Score Altman es valido para PYMES? | Si, con coeficientes ajustados (Altman 2014) | No, fue disenado para manufactura de EE.UU. (Grice 2001) | Usar como heuristica complementaria, no como modelo unico |
| ¿Datos de un pais se generalizan? | Si, patrones financieros son universales | No, contexto institucional y normativo importa | Usar datos nacionales pero reconocer limitaciones de generalizacion |

---

## 6. Estructura Sugerida de Redaccion

### 6.1 Esquema general

```
ESTADO DEL ARTE
|
|-- 6.1 Introduccion del estado del arte
|       (1-2 paginas: alcance, ejes tematicos, metodologia de busqueda)
|
|-- 6.2 Categoria 1: Analisis financiero y riesgo en PYMES
|       (4-6 paginas)
|
|-- 6.3 Categoria 2: Machine Learning para prediccion de riesgo
|       (5-8 paginas)
|
|-- 6.4 Categoria 3: Sistemas web para analisis financiero
|       (3-5 paginas)
|
|-- 6.5 Categoria 4: Contexto colombiano
|       (2-4 paginas)
|
|-- 6.6 Discusion critica y sintesis
|       (2-3 paginas: vacios, tabla comparativa, posicionamiento)
|
|-- 6.7 Cierre
|       (1 pagina: conexion con objetivos de la tesis)
```

### 6.2 Introduccion del estado del arte

**Que incluir:**
- Alcance de la revision (temas cubiertos, periodo, bases de datos consultadas)
- Breve justificacion de por que esos ejes tematicos
- Estructura de la seccion (adelantar al lector que encontrara)

**Ejemplo de redaccion:**

> "El presente capitulo examina la literatura cientifica relevante para tres ejes tematicos que confluyen en esta investigacion: (1) el analisis financiero y la evaluacion de riesgo en PYMES, (2) la aplicacion de tecnicas de Machine Learning para la prediccion de riesgo financiero, y (3) el desarrollo de sistemas web orientados al analisis financiero empresarial. La revision se basa en articulos publicados entre 2015 y 2026, recuperados de las bases de datos Scopus, IEEE Xplore, Web of Science, SciELO y Google Scholar, utilizando las cadenas de busqueda detalladas en el Anexo A. Se priorizaron estudios con revision por pares, con enfasis en investigaciones aplicadas a mercados emergentes y al contexto latinoamericano. La seccion se organiza tematicamente, progresando desde los fundamentos del analisis financiero hasta las aplicaciones tecnologicas, y concluye con una sintesis critica que identifica los vacios de investigacion que este trabajo busca abordar."

### 6.3 Desarrollo de cada categoria

**Estructura interna de cada seccion tematica:**

```
Categoria N: [Titulo]
|
|-- Parrafo de apertura: ¿que pregunta responde esta seccion?
|
|-- Subcategoria N.1: [Subtema]
|   |-- Obras fundacionales / clasicos
|   |-- Avances recientes (2019-2026)
|   |-- Comparacion / sintesis
|
|-- Subcategoria N.2: [Subtema]
|   |-- (misma estructura)
|
|-- Parrafo de transicion: conecta con la siguiente categoria
```

**Ejemplo de parrafo de apertura:**

> "La prediccion de quiebra empresarial ha sido objeto de investigacion desde la decada de 1960, pero su aplicacion especifica a PYMES en economias emergentes es considerablemente mas reciente. Esta seccion examina la evolucion de los modelos predictivos, desde los enfoques estadisticos clasicos hasta las tecnicas contemporaneas de aprendizaje automatico, con atencion particular a los desafios que plantea el contexto de las PYMES."

**Ejemplo de parrafo de transicion:**

> "Los estudios revisados en esta seccion evidencian que los modelos de Machine Learning superan consistentemente a los enfoques clasicos en la prediccion de riesgo financiero. Sin embargo, pocos de estos trabajos traducen sus resultados en herramientas accesibles para los tomadores de decisiones. La siguiente seccion examina los esfuerzos por integrar modelos analiticos en sistemas web funcionales."

### 6.4 Discusion critica y sintesis

Esta es la seccion mas valiosa del estado del arte. Aqui se demuestra pensamiento original.

**Que incluir:**
1. Tabla comparativa de los trabajos mas relevantes (ver seccion 3.3)
2. Identificacion explicita de vacios (ver seccion 5.1)
3. Sintesis de tendencias
4. Posicionamiento de tu investigacion

**Ejemplo de tabla comparativa final:**

| Autor | Ano | Pais | N empresas | Modelo | AUC/Acc | Sistema web | PYMES | Datos NIIF | Panel |
|-------|-----|------|-----------|--------|---------|:-----------:|:-----:|:----------:|:-----:|
| Altman | 1968 | EE.UU. | 66 | MDA | — | No | No | No | No |
| Barboza et al. | 2017 | EE.UU. | 10,000 | Bagging | 0.94 | No | No | No | Si |
| Wang et al. | 2020 | China | 12,000 | XGBoost | 0.89 | No | Si | No | No |
| Lopez | 2019 | Mexico | 200 | LR | 0.78 | No | Si | No | No |
| Perez & Garcia | 2021 | Colombia | 500 | RF | 0.82 | No | Si | No | No |
| **Esta tesis** | **2026** | **Colombia** | **38,245** | **LR, RF, XGB** | **TBD** | **Si** | **Si** | **Si** | **Si** |

**Ejemplo de parrafo de posicionamiento:**

> "Como se evidencia en la Tabla N, la mayoria de estudios sobre prediccion de riesgo en PYMES latinoamericanas utilizan muestras inferiores a 1,000 empresas y no integran los resultados en herramientas de decision. El presente trabajo se distingue en cuatro dimensiones: (1) la escala del dataset (38,245 empresas y 203,104 observaciones empresa-ano), (2) el uso de datos oficiales bajo normas NIIF, (3) la estructura de panel que habilita analisis temporales, y (4) la integracion del modelo predictivo en un sistema web funcional. Esta combinacion no tiene precedente en la literatura revisada para el contexto colombiano."

### 6.5 Cierre del estado del arte

**Que incluir:**
- Recapitulacion de los hallazgos principales de la revision
- Conexion directa con los objetivos de la tesis
- Frase de cierre que justifica la investigacion

**Ejemplo:**

> "La revision de la literatura revela tres hallazgos centrales: (a) los modelos de Machine Learning, particularmente los metodos de ensamble, han demostrado superioridad frente a los modelos estadisticos clasicos en la prediccion de riesgo financiero; (b) la aplicacion de estas tecnicas al segmento PYME en Latinoamerica es incipiente, con datasets pequenos y sin integracion tecnologica; y (c) no se identificaron estudios que combinen analisis financiero bajo NIIF, modelos de Machine Learning y un sistema web de decision para PYMES colombianas. Estos vacios fundamentan los objetivos planteados en la Seccion 1.3 y justifican el diseno metodologico que se presenta en el Capitulo 3."

---

## 7. Estilo Academico y Manejo de Referencias

### 7.1 Normas de citacion

**APA 7 (ciencias sociales, administracion, educacion):**
- Cita en texto: (Altman, 1968) o Altman (1968)
- Dos autores: (Wang & Chen, 2020)
- Tres o mas: (Barboza et al., 2017)
- Cita directa: (Lopez, 2019, p. 45)

**IEEE (ingenieria, computacion, sistemas):**
- Cita en texto: [1], [2], [3]-[5]
- Numeracion secuencial segun orden de aparicion
- Preferido en revistas de ingenieria y conferencias IEEE

> **Para esta tesis**: Si la universidad exige APA, usar APA 7. Si se busca publicar en una revista de ingenieria o computacion, preparar version IEEE tambien.

### 7.2 Citas y parafraseo correcto

**Cita directa (usar con moderacion, maximo 5% del texto):**

> Segun Altman (1968), "the Z-Score model correctly classified 95% of the sample one year prior to bankruptcy" (p. 599).

**Parafraseo correcto:**

> El modelo Z-Score de Altman (1968) logro clasificar correctamente el 95% de las empresas un ano antes de la quiebra, estableciendo un referente para las decadas siguientes de investigacion en prediccion de insolvencia.

**Parafraseo incorrecto (demasiado cercano al original):**

> El modelo Z-Score clasifico correctamente el 95% de la muestra un ano antes de la bancarrota (Altman, 1968).

*(Esto es parafraseo mecanico — solo cambia unas palabras. Debe incluir interpretacion o contexto.)*

### 7.3 Verbos para reportar hallazgos de otros autores

| Categoria | Verbos | Ejemplo |
|-----------|--------|---------|
| **Neutro** | reportar, encontrar, observar, documentar, mostrar | "Wang et al. (2020) encontraron que..." |
| **De acuerdo** | confirmar, demostrar, evidenciar, validar, corroborar | "Los resultados de Martinez (2020) confirman que..." |
| **Cauteloso** | sugerir, indicar, plantear, proponer, estimar | "Lopez (2019) sugiere que..." |
| **Critico** | asumir, afirmar sin evidencia, omitir, simplificar, ignorar | "Rodriguez (2021) asume que..., sin considerar..." |
| **Evolutivo** | extender, ampliar, mejorar, superar, actualizar | "Chen (2022) extiende el enfoque de Wang (2020) al incorporar..." |

### 7.4 Coherencia y conectores logicos

**Conectores de adicion:**
- Ademas, asimismo, de igual modo, en la misma linea, complementariamente

**Conectores de contraste:**
- Sin embargo, no obstante, en contraste, por el contrario, a pesar de

**Conectores de causa:**
- Debido a, como consecuencia de, dado que, en virtud de, a raiz de

**Conectores de concesion:**
- Aunque, si bien, a pesar de que, aun cuando, pese a

**Conectores de sintesis:**
- En sintesis, en suma, a partir de lo expuesto, como se ha mostrado

### 7.5 Errores frecuentes a evitar

| Error | Ejemplo | Correccion |
|-------|---------|-----------|
| Citar sin analizar | "Segun X (2020)... Segun Y (2021)... Segun Z (2022)..." | Agrupar autores por hallazgo comun |
| Usar "se ha demostrado que" sin cita | "Se ha demostrado que XGBoost es el mejor modelo" | ¿Quien lo demostro? Citar la fuente |
| Exceso de citas directas | Parrafos enteros entre comillas | Parafrasear e interpretar |
| Citar fuentes no academicas | "Segun la pagina web de IBM..." | Buscar el paper academico detras del producto |
| No citar articulos recientes | Ultima cita de 2018 | Los revisores buscan referencias de los ultimos 3-5 anos |
| Usar "et al." con dos autores | "Wang et al. (2020)" cuando solo son Wang y Chen | APA: "Wang y Chen (2020)" con dos autores |

---

## 8. Conexion con Marco Teorico y Objetivos

### 8.1 Diferencia entre estado del arte y marco teorico

| Aspecto | Estado del arte | Marco teorico |
|---------|----------------|---------------|
| **Pregunta que responde** | ¿Que se ha investigado sobre este tema? | ¿Que conceptos y teorias fundamentan mi trabajo? |
| **Contenido** | Trabajos empiricos, resultados, metodologias de otros autores | Definiciones, teorias, modelos conceptuales, normativas |
| **Temporalidad** | Enfasis en lo reciente (ultimos 5-10 anos) | Puede incluir teorias clasicas atemporales |
| **Tono** | Critico-comparativo | Descriptivo-explicativo |
| **Resultado** | Vacios de investigacion | Base conceptual para variables y constructos |

### 8.2 Como se conectan

```
MARCO TEORICO                    ESTADO DEL ARTE
(Que es...)                      (Que se ha hecho con...)
|                                |
| Define "riesgo financiero"     | Revisa estudios que predicen riesgo
| Define "indicador financiero"  | Compara que indicadores usan otros
| Explica NIIF para PYMES        | Busca estudios que usan datos NIIF
| Teorias de ML supervisado      | Compara resultados de LR, RF, XGB
|                                |
|___________ Ambos fundamentan __|
             |
             v
        OBJETIVOS DE LA TESIS
        (Que voy a hacer yo)
```

### 8.3 Como conectar estado del arte con objetivos

Cada objetivo de la tesis debe tener respaldo en el estado del arte. Ejemplo:

| Objetivo especifico | Sustento en el estado del arte |
|---------------------|-------------------------------|
| Calcular indicadores financieros para PYMES | "Los indicadores de liquidez, solvencia y rentabilidad son los mas utilizados en la literatura (Autor1, Autor2, Autor3)" |
| Construir un modelo ML de clasificacion de riesgo | "XGBoost y RF han mostrado superioridad en datos tabulares financieros (Autor4, Autor5), pero no se han aplicado a datos NIIF colombianos" |
| Desarrollar un sistema web | "Los estudios revisados no integran modelos ML en herramientas de decision accesibles para PYMES (vacio identificado)" |
| Validar con empresas de Ibague | "Los estudios nacionales existentes (Autor6) usan muestras limitadas; este trabajo amplia la escala y valida localmente" |

---

## 9. Checklist Final antes de Entregar

### 9.1 Contenido

- [ ] ¿Cada eje tematico tiene al menos 8-10 referencias?
- [ ] ¿Se incluyen tanto fuentes clasicas/seminales como recientes (ultimos 3 anos)?
- [ ] ¿Se incluyen al menos 3-5 referencias de contexto latinoamericano/colombiano?
- [ ] ¿La tabla comparativa incluye al menos 10-15 trabajos relevantes?
- [ ] ¿Se identifican al menos 3 vacios de investigacion explicitos?
- [ ] ¿Se posiciona la tesis frente a la literatura existente?
- [ ] ¿Cada seccion tiene parrafo de apertura y transicion al siguiente?

### 9.2 Calidad academica

- [ ] ¿Hay analisis critico y no solo resumen descriptivo?
- [ ] ¿Se agrupan autores por hallazgo/tendencia en lugar de uno por parrafo?
- [ ] ¿Se evitan generalizaciones sin respaldo? ("se ha demostrado...", "es bien sabido...")
- [ ] ¿Las citas estan en formato consistente (APA 7 o IEEE)?
- [ ] ¿Se usan mas parafraseos que citas directas?
- [ ] ¿Se evitan fuentes no academicas (blogs, Wikipedia, paginas comerciales)?

### 9.3 Para publicacion en revista

- [ ] ¿La revision incluye articulos de las revistas donde se planea publicar?
- [ ] ¿Se sigue el formato de citacion de la revista destino?
- [ ] ¿Se incluyen los trabajos mas citados del campo? (verificar con Scopus/Google Scholar)
- [ ] ¿Se citan al menos 2-3 articulos de los ultimos 2 anos?
- [ ] ¿La seccion tiene la extension adecuada para la revista? (tipicamente 2-4 paginas)
- [ ] ¿Se comparan resultados cuantitativos (AUC, accuracy, F1) cuando estan disponibles?

### 9.4 Conexion con la tesis

- [ ] ¿Cada objetivo tiene al menos una referencia que lo sustenta?
- [ ] ¿Los vacios identificados coinciden con lo que la tesis propone resolver?
- [ ] ¿Las decisiones metodologicas (modelo ML, stack tecnologico) estan respaldadas por la literatura?
- [ ] ¿El estado del arte conecta fluidamente con el marco teorico (sin repetir contenido)?

---

## 10. Recursos Complementarios

### 10.1 Articulos seminales para esta tesis (punto de partida)

| Referencia sugerida | Tema | Por que es relevante |
|---------------------|------|---------------------|
| Altman, E. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance* | Z-Score | Fundamento del etiquetado de riesgo |
| Ohlson, J. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research* | O-Score | Alternativa clasica con regresion logistica |
| Barboza, F. et al. (2017). Machine learning models and bankruptcy prediction. *Expert Systems with Applications* | ML vs clasicos | Comparativa amplia de modelos |
| Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD* | XGBoost | Fundamento tecnico del modelo propuesto |
| Breiman, L. (2001). Random Forests. *Machine Learning* | Random Forest | Fundamento tecnico |
| Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS* | SHAP | Interpretabilidad de modelos |

> **Importante**: Estos son puntos de partida. La busqueda sistematica en Scopus/WoS revelara los articulos mas recientes y relevantes para cada eje.

### 10.2 Estructura sugerida de carpetas para organizar la busqueda

```
literature/
|-- 01_analisis_financiero/
|   |-- clasicos/           (Altman, Ohlson, Zmijewski)
|   |-- PYMES/              (estudios en PYMES)
|   |-- colombia/            (contexto local)
|
|-- 02_machine_learning/
|   |-- comparativas/        (ML vs modelos clasicos)
|   |-- XGBoost_RF/          (ensambles)
|   |-- interpretabilidad/   (SHAP, LIME)
|
|-- 03_sistemas_web/
|   |-- dashboards/
|   |-- integracion_ML/
|
|-- 04_contexto/
|   |-- NIIF_Colombia/
|   |-- PYMES_Latinoamerica/
|
|-- matriz_literatura.xlsx   (mapeo de toda la literatura)
|-- tabla_comparativa.xlsx   (tabla final para el documento)
```
