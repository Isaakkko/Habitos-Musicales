# LIBRERIAS
import pandas as pd
import plotly.express as px


class VisualizacionMusicStreaming:

    def __init__(self, datos):
        # acepta un dataframe o la ruta del csv limpio
        if isinstance(datos, pd.DataFrame):
            self.df = datos.copy()
        else:
            self.df = pd.read_csv(datos)

    #  MINUTOS DE ESCUCHA SEGUN LA PLATAFORMA
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

    #  CANCIONES ESCUCHADAS SEGUN EL ESTADO DE ANIMO
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

    #  RELACION ENTRE MINUTOS DE ESCUCHA Y CANCIONES POR DIA
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