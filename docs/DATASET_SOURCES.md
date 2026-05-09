# Fuentes de los Datasets

Los archivos crudos de la carpeta `dataset/` no se versionan en este repositorio porque dos de ellos exceden el límite de **2 GB por archivo** de Git LFS en GitHub. Este documento lista los nombres exactos y el origen para poder redescargarlos.

## Archivos esperados en `dataset/`

Total aproximado: **~9.2 GB**

| # | Nombre exacto del archivo | Tamaño | Origen |
|---|---------------------------|--------|--------|
| 1 | `Estados_Financieros_NIIF-_Estado_de_Situación_Financiera_20260203.csv` | 4.10 GB | Supersociedades — Portal de Datos Abiertos |
| 2 | `Estados_Financieros_NIIF-_Carátula_20260204.csv` | 2.07 GB | Supersociedades — Portal de Datos Abiertos |
| 3 | `Estados_Financieros_NIIF-_Estado_de_Resultado_Integral_20260204.csv` | 1.57 GB | Supersociedades — Portal de Datos Abiertos |
| 4 | `Estados_Financieros_NIIF-_Estado_de_Flujo_Efectivo_20260204.csv` | 1.43 GB | Supersociedades — Portal de Datos Abiertos |
| 5 | `BASE_DE_DATOS_DE_EMPRESAS_Y_O_ENTIDADES_ACTIVAS_-_JURISDICCIÓN_CÁMARA_DE_COMERCIO_DE_IBAGUÉ_-_CORTE_A_31_DE_DICIEMBRE_DE_2025_20260207.csv` | 25 MB | Cámara de Comercio de Ibagué — Datos Abiertos |

## Dónde redescargar

### 1–4. Estados Financieros NIIF (SIREM)

Fuente oficial: **Superintendencia de Sociedades de Colombia (Supersociedades)**, sistema **SIREM** (Sistema de Información y Riesgo Empresarial).

- Portal de Datos Abiertos del Gobierno de Colombia: https://www.datos.gov.co
- Portal Supersociedades: https://www.supersociedades.gov.co

**Términos de búsqueda recomendados en datos.gov.co:**
- "Estados Financieros NIIF Carátula"
- "Estados Financieros NIIF Estado de Situación Financiera"
- "Estados Financieros NIIF Estado de Resultado Integral"
- "Estados Financieros NIIF Estado de Flujo de Efectivo"

El sufijo `_20260204` / `_20260203` corresponde a la fecha de corte del snapshot publicado (formato `YYYYMMDD`). Las versiones futuras tendrán otro sufijo; los archivos se actualizan periódicamente.

### 5. Empresas activas — Cámara de Comercio de Ibagué

Fuente oficial: **Cámara de Comercio de Ibagué**, base de datos pública de empresas activas en su jurisdicción.

- Sitio: https://www.ccibague.org
- También suele estar replicado en https://www.datos.gov.co

Búsqueda: "Cámara de Comercio Ibagué empresas activas".

## Notas

- Encoding original: UTF-8 (algunos `CONCEPTO` traen artefactos como `�` para caracteres acentuados — manejarlo en lectura).
- Formato: CSV vertical/long (una fila por par concepto-valor, no una fila por empresa).
- Columnas comunes: `NIT`, `PUNTO_ENTRADA`, `TAXONOMIA`, `CONCEPTO`, `PERIODO`, `VALOR`.
- Para más detalle de estructura y conceptos financieros, ver `DATASETS_SIREM.md`.

## Procesados

Los archivos consolidados en la carpeta `data/` (`colombia_*_pymes.csv`) sí están versionados en este repositorio vía Git LFS, ya que pesan menos de 2 GB cada uno. Son derivados procesados de los archivos crudos anteriores.
