# LIBRERIAS
import os
import pandas as pd
import plotly.express as px
from pathlib import Path

# raiz del repositorio (se calcula desde la ubicacion de este archivo)
RAIZ = Path(__file__).resolve().parents[2]


class EDAMusicStreaming:

    def __init__(self, ruta_csv=RAIZ / "DATA" / "RAW" / "music_streaming_habits_2026.csv"):
        # ruta del archivo original y dataframe donde se guardan los datos
        self.ruta_csv = ruta_csv
        self.df = None

        # variables que se usan en los graficos
        self.variables_numericas = ["edad", "minutosXdia", "cancionesXdia", "playlist"]
        self.variables_categoricas = [
            "pais", "plataforma", "suscripcion", "genero_favorito",
            "top_artista", "estado_animo", "podcast"
        ]

    # LECTURA DEL DATASET
    def cargar_datos(self):
        # lee el csv y muestra las primeras filas y la dimension
        self.df = pd.read_csv(self.ruta_csv)
        print(self.df.head(5))
        print(self.df.shape)

    # se eliminan las columnas que no son necesarias para este analisis
    def eliminar_columnas(self):
        self.df = self.df.drop(columns=[
            "listener_id",
            "skip_rate_pct",
            "discover_weekly_user",
            "uses_offline_mode"])
        print(self.df.head(5))

    # se renombran las columnas al español
    def renombrar_columnas(self):
        self.df = self.df.rename(columns={
            "age": "edad",
            "country": "pais",
            "platform": "plataforma",
            "subscription": "suscripcion",
            "top_genre": "genero_favorito",
            "top_artist": "top_artista",
            "daily_listening_minutes": "minutosXdia",
            "songs_per_day": "cancionesXdia",
            "playlists_count": "playlist",
            "top_mood": "estado_animo",
            "podcasts_too": "podcast"})
        print(self.df.head(5))

    # revision de valores nulos y tipos de datos
    def revisar_datos(self):
        print(self.df.isna().sum())
        self.df.info()

    # se convierte podcast a entero que acepta nulos
    def convertir_tipos(self):
        self.df["podcast"] = self.df["podcast"].astype("Int64")
        print(self.df.head(5))

    # resumen estadistico de las variables numericas
    def describir_datos(self):
        print(self.df.describe())

    # guardar dataset ya limpio
    def guardar_dataset(self, ruta_salida=RAIZ / "DATA" / "PROCESSED" / "music_streaming_habits_limpio.csv"):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        self.df.to_csv(ruta_salida, index=False)

    # 1. DISTRIBUCION DE LAS VARIABLES NUMERICAS
    def graficar_distribucion_numericas(self):
        for variable in self.variables_numericas:
            fig = px.histogram(
                self.df,
                x=variable,
                nbins=30,
                marginal="box",
                title=f"Distribución de {variable}",
                labels={variable: variable.replace("_", " ").title(), "count": "Cantidad de personas"},
                color_discrete_sequence=["#4169A1"]
            )
            fig.update_layout(template="plotly_white", hovermode="x unified")
            fig.show()

            # estadisticos para la interpretacion
            promedio = self.df[variable].mean()
            mediana = self.df[variable].median()
            minimo = self.df[variable].min()
            maximo = self.df[variable].max()

            print(
                f"Interpretación: Para {variable}, el promedio es {promedio:.2f} "
                f"y la mediana es {mediana:.2f}. Los valores se encuentran entre "
                f"{minimo} y {maximo}. El gráfico permite identificar en qué rangos "
                f"se concentra la mayoría de los registros y observar posibles valores extremos."
            )

    # 2. COMPARACION DE LAS VARIABLES NUMERICAS
    def graficar_comparacion_numericas(self):
        # se pasan las variables a formato largo para comparar en un solo grafico
        datos_numericos = self.df[self.variables_numericas].melt(
            var_name="Variable",
            value_name="Valor"
        )

        fig = px.box(
            datos_numericos,
            x="Variable",
            y="Valor",
            color="Variable",
            points="outliers",
            title="Comparación de las variables numéricas"
        )
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.show()

        print(
            "Interpretación: El gráfico permite comparar la variabilidad de las edades, "
            "los minutos de escucha, las canciones por día y las listas de reproducción. "
            "Los puntos que aparecen fuera de los bigotes representan posibles valores "
            "atípicos. Cada variable tiene una escala diferente, por lo que se debe "
            "interpretar considerando sus propias unidades."
        )

    # 3. FRECUENCIA DE LAS VARIABLES CATEGORICAS
    def graficar_frecuencia_categoricas(self):
        for variable in self.variables_categoricas:
            # conteo de registros por categoria
            frecuencias = self.df[variable].value_counts().reset_index()
            frecuencias.columns = [variable, "cantidad"]
            frecuencias = frecuencias.sort_values("cantidad", ascending=False)

            fig = px.bar(
                frecuencias,
                x=variable,
                y="cantidad",
                title=f"Cantidad de registros por {variable}",
                labels={variable: variable.replace("_", " ").title(),
                        "cantidad": "Cantidad de personas"},
                color="cantidad",
                color_continuous_scale="Blues",
                hover_data={"cantidad": True}
            )
            fig.update_layout(template="plotly_white", xaxis_tickangle=-35)
            fig.show()

            # categoria mas frecuente para la interpretacion
            categoria_mayor = frecuencias.iloc[0][variable]
            cantidad_mayor = int(frecuencias.iloc[0]["cantidad"])
            porcentaje = cantidad_mayor / len(self.df) * 100

            print(
                f"Interpretación: La categoría más frecuente en {variable} es "
                f"'{categoria_mayor}', con {cantidad_mayor} registros "
                f"({porcentaje:.1f}% del total). Esto permite reconocer cuál opción "
                f"se presenta con mayor frecuencia en el conjunto de datos."
            )

    # 4. MINUTOS DE ESCUCHA SEGUN LA PLATAFORMA
    def graficar_minutos_por_plataforma(self):
        fig = px.box(
            self.df,
            x="plataforma",
            y="minutosXdia",
            color="plataforma",
            points="outliers",
            title="Minutos diarios de escucha según plataforma",
            labels={"plataforma": "Plataforma", "minutosXdia": "Minutos por día"}
        )
        fig.update_layout(template="plotly_white", showlegend=False, xaxis_tickangle=-25)
        fig.show()

        # plataforma con el promedio mas alto
        promedios_plataforma = self.df.groupby("plataforma")["minutosXdia"].mean().sort_values(ascending=False)
        print(
            f"Interpretación: La plataforma con el promedio más alto de minutos diarios "
            f"es {promedios_plataforma.index[0]}, con {promedios_plataforma.iloc[0]:.1f} "
            f"minutos. La comparación describe los hábitos registrados en cada plataforma, "
            f"pero no demuestra que la plataforma sea la causa de escuchar más música."
        )

    # 5. CANCIONES ESCUCHADAS SEGUN EL ESTADO DE ANIMO
    def graficar_canciones_por_animo(self):
        fig = px.box(
            self.df,
            x="estado_animo",
            y="cancionesXdia",
            color="estado_animo",
            points="outliers",
            title="Canciones por día según estado de ánimo",
            labels={"estado_animo": "Estado de ánimo", "cancionesXdia": "Canciones por día"}
        )
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.show()

        # estado de animo con el promedio mas alto
        promedios_animo = self.df.groupby("estado_animo")["cancionesXdia"].mean().sort_values(ascending=False)
        print(
            f"Interpretación: El estado de ánimo con el promedio más alto de canciones "
            f"escuchadas por día es {promedios_animo.index[0]}, con "
            f"{promedios_animo.iloc[0]:.1f} canciones. El gráfico permite observar "
            f"si existen diferencias entre los grupos."
        )

    # 6. RELACION ENTRE MINUTOS DE ESCUCHA Y CANCIONES POR DIA
    def graficar_minutos_vs_canciones(self):
        fig = px.scatter(
            self.df,
            x="minutosXdia",
            y="cancionesXdia",
            color="plataforma",
            hover_data=["edad", "genero_favorito", "estado_animo"],
            title="Relación entre minutos de escucha y canciones por día",
            labels={"minutosXdia": "Minutos de escucha por día",
                    "cancionesXdia": "Canciones por día"}
        )
        fig.update_layout(template="plotly_white")
        fig.show()

        # correlacion entre ambas variables
        correlacion = self.df["minutosXdia"].corr(self.df["cancionesXdia"])

        # se clasifica la fuerza de la relacion
        if correlacion > 0.5:
            nivel = "una relación positiva marcada"
        elif correlacion > 0.2:
            nivel = "una relación positiva moderada"
        elif correlacion >= -0.2:
            nivel = "una relación lineal débil o prácticamente inexistente"
        elif correlacion >= -0.5:
            nivel = "una relación negativa moderada"
        else:
            nivel = "una relación negativa marcada"

        print(
            f"Interpretación: La correlación entre los minutos de escucha y las canciones "
            f"por día es {correlacion:.2f}, lo que indica {nivel}. Esto permite observar "
            f"cómo se relacionan ambas variables, pero no significa que una cause la otra."
        )

    # 7. MATRIZ DE CORRELACION
    def graficar_matriz_correlacion(self):
        matriz_correlacion = self.df[self.variables_numericas].corr()

        fig = px.imshow(
            matriz_correlacion,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de correlación de las variables numéricas"
        )
        fig.update_layout(template="plotly_white")
        fig.show()

        print(
            "Interpretación: La matriz permite identificar qué variables presentan "
            "relaciones lineales más fuertes. La diagonal muestra valores de 1 porque "
            "cada variable tiene una correlación perfecta consigo misma. Las correlaciones "
            "no representan necesariamente relaciones de causa y efecto."
        )

