# Hábitos Musicales

Análisis de datos de streaming musical con la metodología **CRISP-DM**: exploración, preprocesamiento automatizado mediante un pipeline orientado a objetos (POO) en Python, reducción de dimensionalidad, segmentación de usuarios y reglas de asociación para un sistema de recomendación.

## Contenido

- [Contexto académico](#contexto-académico)
- [Problema y objetivos](#problema-y-objetivos)
- [Dataset](#dataset)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Autores](#autores)

## Contexto académico

| | |
|---|---|
| Institución | Colegio Universitario de Cartago |
| Carrera | Big Data |
| Curso | Minería de Datos II |
| Profesor | Ronald Córdoba Vega |

## Problema y objetivos

Una plataforma de streaming musical (al estilo de Spotify o Apple Music) genera grandes volúmenes de datos de escucha. Este proyecto busca convertirlos en conocimiento útil para personalizar la experiencia del usuario, mejorar las recomendaciones y reducir el abandono de la plataforma.

**Objetivos específicos**

1. Explorar y describir los hábitos de escucha de los 4,000 usuarios del dataset.
2. Automatizar la ingesta, limpieza y transformación de los datos con un pipeline modular en POO.
3. Reducir la dimensionalidad (ACP, t-SNE y UMAP) para visualizar y comprender la estructura de los datos.
4. Segmentar a los usuarios según su comportamiento.
5. Descubrir reglas de asociación (Apriori, FP-Growth, Eclat) para sugerir contenido.

**Criterios de éxito**

| Tipo | Criterio |
|---|---|
| Cualitativo | Recomendaciones más afines para el usuario y mejor segmentación comercial |
| Cuantitativo | Mayor tiempo promedio de escucha diaria, mayor aceptación de recomendaciones y menor tasa de abandono |

## Dataset

- **Fuente:** [Music Streaming Habits 2026](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026) (Kaggle).
- **Tamaño:** 4,000 registros y 15 variables.
- **Formato:** CSV, en `DATA/RAW/`. Las versiones depuradas quedan en `DATA/PROCESSED/`.

| Variable | Tipo | Descripción |
|---|---|---|
| `listener_id` | Identificador | Identificador único del usuario |
| `age` | Numérica | Edad del usuario |
| `country` | Categórica | País de residencia |
| `platform` | Categórica | Plataforma de streaming utilizada |
| `subscription` | Categórica | Tipo de suscripción (Free, Premium) |
| `top_genre` | Categórica | Género musical más escuchado |
| `top_artist` | Categórica | Artista favorito |
| `daily_listening_minutes` | Numérica | Minutos de escucha por día |
| `songs_per_day` | Numérica | Canciones reproducidas por día |
| `playlists_count` | Numérica | Número de listas de reproducción creadas |
| `skip_rate_pct` | Numérica (porcentaje) | Porcentaje de canciones omitidas |
| `discover_weekly_user` | Booleana | Usa el descubrimiento semanal |
| `top_mood` | Categórica | Estado de ánimo predominante al escuchar |
| `uses_offline_mode` | Booleana | Usa el modo sin conexión |
| `podcasts_too` | Booleana | También consume podcasts |

## Estructura del repositorio

```text
Habitos-Musicales/
├── CLASES_POO/              # Pipeline orientado a objetos
│   ├── EDA.py               # Análisis exploratorio
│   ├── VIZUALIZADOR.py      # Visualizaciones generales
│   └── POO.py               # Clases de ACP, t-SNE y UMAP
├── CRISP-DM/
│   └── CRISP-DM.docx        # Documento de la metodología (fases 1 y 2)
├── DATA/
│   ├── RAW/                 # Datos originales
│   └── PROCESSED/           # Datos limpios y transformados
├── HTML/
│   ├── PROYECTO.ipynb       # Notebook principal
│   └── PROYECTO.html        # Informe técnico
└── MODELOS/
    ├── CLUSTERING/          # CLUSTERING.py
    ├── REGLAS DE ASOCIACION/# ASOCIACION.py
    ├── TSNE/                # TSNE.py y TSNE_Analisis.ipynb
    └── UMAP/                # MODELO_UMAP.py y UMAP.ipynb
```

## Metodología

El proyecto sigue las fases de CRISP-DM.

| Fase | Qué se hace | Dónde |
|---|---|---|
| 1. Comprensión del negocio | Objetivos, criterios de éxito, análisis costo-beneficio y matriz de riesgos | `CRISP-DM/CRISP-DM.docx` |
| 2. Comprensión de los datos | Descripción de variables, estadísticas descriptivas, gráficos y revisión de calidad (nulos y outliers) | `CLASES_POO/EDA.py`, `CLASES_POO/VIZUALIZADOR.py` |
| 3. Preparación de los datos | Limpieza y transformación | `DATA/PROCESSED/` |
| 4. Modelado | ACP, t-SNE, UMAP, clustering y reglas de asociación | `CLASES_POO/POO.py`, `MODELOS/` |
| 5. Evaluación y presentación | Interpretación de resultados e informe técnico | `HTML/PROYECTO.html` |

**Análisis exploratorio.** Estadísticas descriptivas (media, desviación estándar, cuartiles), boxplots de las variables numéricas, barras de las variables categóricas, minutos por día según plataforma, canciones por día según estado de ánimo y dispersión de minutos contra canciones por día con su correlación.

## Autores

- Isaac Ulloa Calvo
- Mariel Rodríguez Arguello
- Wedell Orozco Gonzáles
- Mónica Mendoza Morales
- Tiffany Méndez Quirós
