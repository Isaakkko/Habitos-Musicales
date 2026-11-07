# Habitos-Musicales

Se realiza un análisis exploratorio, métodos de preprocesamiento automatizados mediante un pipeline orientado a objetos (POO) y modelos de Machine Learning para el descubrimiento de patrones y comportamientos en los datos de streaming musical, con el fin de darles una interpretación y valor estratégico.

Componentes principales del proyecto:

**Pipeline POO:** Arquitectura modular y reutilizable para la ingesta, limpieza y transformación de los datos.

**Análisis Exploratorio (EDA):** Estadísticas descriptivas y visualizaciones para comprender las distribuciones de los 4,000 registros y 15 variables del dataset.

**Minería de Datos y Modelado:** Aplicación de algoritmos para la generación de reglas de asociación (recomendación).


#### **Comprensión del negocio**

El objetivo central en esta fase es traducir los problemas o necesidades de una plataforma de streaming de música (estilo Spotify o Apple Music) en metas concretas de Minería de Datos (Data Mining) y en un plan de trabajo.

**1.Determinar los objetivos del negocio:**

* **Problema a resolver:** Identificar patrones de consumo, preferencias musicales y comportamiento de escucha de los usuarios para personalizar la experiencia, mejorar los sistemas de recomendación y optimizar la retención.

* **Uso del Data Mining:** Los datos de reproducciones e interacciones son masivos y complejos. La minería de datos permite extraer patrones (como hábitos de escucha por ciertos minutos o géneros complementarios) y agrupar usuarios con comportamientos similares.
  
* **Como medir el éxito:** 

**Cualitativos:** Recomendaciones más afines percibidas por el usuario y mejor segmentación comercial. 

**Cuantitativos:** Incrementar el tiempo promedio de escucha diaria, aumentar la tasa de aceptación de recomendaciones y reducir la tasa de abandono de la plataforma. 

**2.Valoración de la Situación:**

* **Conocimiento acerca del problema:** Se conoce que el comportamiento de escucha varía según el momento del día, el contexto del usuario (trabajo, ejercicio, relax), la edad y el tipo de dispositivo utilizado.

* **Contar con la cantidad requerida de datos:** Se dispone de un total de 4000 filas y 15 columnas.

* **Relación coste-beneficio de Data Mining:** La relación es alta. La optimización del pipeline de datos y los modelos de segmetación/recomendación maximizan el valor de vida del cliente frente a un costo computacional y de desarrollo acotado.

**3.Determinar los Objetivos de Data Mining:**

**Reglas de Asociación:** Descubrir patrones frecuentes usando algoritmos como A priori, FP-Growth y Eclat para sugerir contenido a los clientes.

**4.Realizar el Plan del Proyecto:**

* **Ingreso de datos:** Ingesta, limpieza y transformación mediante clases en Python
* **Fase Exploratoria:** Análisis descriptivo, histogramas y correlaciones.
* **Fase de Modelado:** Reglas de asociación y segmentación.
* **Exposición de resultados:** Interpretación de los resultados y conclusiones.



#### **Comprensión de los Datos**

**1.Recolectar los Datos Iniciales:** Fuente de origen: Repositorio público de Kaggle (Music Streaming Habits 2026).

&#x20;

link: https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026



Formato y ubicación: Archivo csv ubicado en carpeta data/raw/ la cual luego de realizar la limpiza correspondiente estará en data/processed/



**2.Descripción de los Datos:** 4000 filas y 15 columnas

**1.	listener_id (Numérico / Identificador):** Identificador único para cada usuario o escucha en la plataforma. 

**2.	age (Numérico):** Edad del usuario. 

**3.	country (Texto / Categórico):** País de residencia o procedencia del usuario. 

**4.	platform (Texto / Categórico):** Plataforma de streaming de música utilizada (ej. Spotify, Apple Music, etc.). 

**5.	subscription (Texto / Categórico):** Tipo de suscripción del usuario (ej. Free, Premium). 

**6.	top_genre (Texto / Categórico):** Género musical preferido o más escuchado por el usuario. 

**7.	top_artist (Texto / Categórico):** Artista favorito del usuario. 

**8.	daily_listening_minutes (Numérico):** Tiempo total de escucha diaria expresado en minutos. 

**9.	songs_per_day (Numérico):** Cantidad promedio de canciones reproducidas por día. 

**10.	playlists_count (Numérico):** Número de listas de reproducción creadas por el usuario. 

**11.	skip_rate_pct (Texto / Numérico porcentual):** Porcentaje o tasa de omisión (skip rate) de canciones. 

**12.	discover_weekly_user (Booleano):** Indicador de si el usuario hace uso de la función de descubrimiento semanal (TRUE / FALSE). 

**13.	top_mood (Texto / Categórico):** Estado de ánimo predominante o asociado a las sesiones de escucha del usuario (ej. Sleep, activo, relajado). 

**14.	uses_offline_mode (Booleano):** Indica si el usuario utiliza el modo sin conexión a internet (TRUE / FALSE). 

**15.	podcasts_too (Booleano):** Indica si el usuario también consume podcasts en la plataforma (TRUE / FALSE). 




**3.Exploración de los Datos (EDA):** 

* **Estadísticas:** Se usa la aplicación de estadísticas para obtener las medidas de tendencia central (como la media), de dispersión (desviación estándar, mínimos, máximos) y de posición (cuartiles del 25%, 50% o mediana, y 75%) para revelar las propiedades iniciales del dataset.

* **Gráficos:** 

  **Boxplot comparativo de las variables numéricas:** Permite colocar las variables numéricas lado a lado en cajas para mostrar la mediana y la dispersión de cada una, identificando posibles valores atípicos (outliers) fuera de los bigotes e interpretándolas por separado debido a sus diferentes escalas.

  **Gráficos de barras de variables categóricas:** Diseñados para contar cuántas personas se registran en cada categoría (país, plataforma, suscripción, género favorito, artista, estado de ánimo y podcast), identificando cuáles son las más frecuentes o populares.
  
  **Boxplot de minutos por día según la plataforma:** Se puede utilizar para comparar el tiempo de escucha entre los usuarios de cada plataforma, mostrando la mediana, la variación interna y los casos atípicos para describir hábitos de consumo.
  
  **Boxplot de canciones por día según el estado de ánimo:** Permite revisar si la cantidad de canciones reproducidas varía dependiendo del estado de ánimo del usuario (ej. feliz, triste, enérgico) y cuál presenta el promedio más alto.
  
  **Diagrama de dispersión de minutos vs. canciones por día:** Visualización donde cada punto representa a un usuario con colores por plataforma, sirviendo para observar si quienes escuchan más minutos también reproducen más canciones, apoyándose en el cálculo de una correlación clasificada como débil, moderada o marcada.


**4.Verificar la Calidad de los Datos:** 

Identificar cuáles columnas presentan datos faltantes.

Detectar los outliers.

