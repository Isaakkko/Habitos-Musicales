# Habitos-Musicales

Se realiza un analisis exploratorio, metodos y modelos de ML para el descubrimiento de patrones y comportamientos en los datos, con el fin de darles una interpretación y valor



#### **Comprensión del negocio**



**1.Determinar los objetivos del negocio:**

* **Problema a resolver:** La industria de plataformas digital musical enfrenta una alta tasa de cancelación de suscripciones y una baja retención de usuarios.
* **Como medir el éxito:** ------------



**2.Valoración de la Situación:**

Se dispone de registros históricos de hábitos de streaming de usuarios activos capturados durante el año 2026 la cual contiene distintas columnas de interés.



Los datos son accesibles y están disponibles mediante un csv



**3.Determinar los Objetivos de Data Mining:**

Implementar reglas de asociación como algoritmo A priori, FP-Growth y Eclat para identificar patrones frecuentes



Implementar modelos supervisados o no supervisados para segmentar a la audiencia en conglomerados homogéneos.



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

* User\_ID: Identificador único del usuario (variable discreta / nominal).
* Age: Edad del usuario (numérica entera).
* Gender: Género del usuario (categórica nominal).
* Country: País de origen (categórica nominal).
* Preferred\_Platform: Plataforma habitual de reproducción (Spotify, Apple Music, YouTube Music, etc. — nominal).
* Subscription\_Plan: Tipo de cuenta (Free, Premium, Family — ordinal/nominal).
* Listening\_Hours\_Per\_Day: Horas promedio diarias de escucha (numérica continua).
* Time\_of\_Day: Momento de mayor actividad (Morning, Afternoon, Night — categórica).
* Top\_Genres: Género principal reproducido (Rock, Pop, Reggaeton, Jazz, etc. — nominal).
* Device\_Used: Dispositivo principal de acceso (Smartphone, PC, Smart Speaker, etc. — nominal).



**3.Exploración de los Datos (EDA):** ------Que se va implementar------------------



**4.Verificar la Calidad de los Datos:** -------Que proceso se piensa realizar y resultados(ejemplo cuantos NaN)--------------

