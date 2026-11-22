# LIBRERIAS
import os
import pandas as pd
import plotly.express as px


class EDA:

    def __init__(self, ruta_csv):
        # ruta del archivo original y dataframe donde se guardan los datos
        self.ruta_csv = ruta_csv
        self.df = None

    # ---------- CARGA Y GUARDADO ----------

    def cargar_datos(self):
        self.df = pd.read_csv(self.ruta_csv)
        print(self.df.head(5))
        print(self.df.shape)

    def guardar_dataset(self, ruta_salida):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        self.df.to_csv(ruta_salida, index=False)

    # ---------- DETECCION DE VARIABLES ----------

    # si no se pasan variables, se detectan segun el tipo de dato
    def _numericas(self, variables=None):
        if variables is None:
            return self.df.select_dtypes(include="number").columns.tolist()
        return variables

    def _categoricas(self, variables=None):
        if variables is None:
            return self.df.select_dtypes(exclude="number").columns.tolist()
        return variables

    # ---------- REVISION Y LIMPIEZA ----------

    # tipo, nulos, porcentaje de nulos y valores unicos por columna
    def resumen_calidad(self):
        resumen = pd.DataFrame({
            "tipo": self.df.dtypes,
            "nulos": self.df.isna().sum(),
            "% nulos": (self.df.isna().mean() * 100).round(2),
            "unicos": self.df.nunique()
        })
        print(resumen)
        print(f"\nFilas duplicadas: {self.df.duplicated().sum()}")

    def eliminar_duplicados(self):
        antes = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Se eliminaron {antes - len(self.df)} filas duplicadas")

    # ---------- DESCRIPCION ----------

    def describir_datos(self):
        print(self.df.describe())

    def describir_categoricas(self, variables=None):
        print(self.df[self._categoricas(variables)].describe())

    def detectar_outliers(self, variables=None):
        resultados = []
        for variable in self._numericas(variables):
            q1 = self.df[variable].quantile(0.25)
            q3 = self.df[variable].quantile(0.75)
            iqr = q3 - q1
            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr
            n = ((self.df[variable] < limite_inf) | (self.df[variable] > limite_sup)).sum()
            resultados.append({
                "variable": variable,
                "limite_inferior": round(limite_inf, 2),
                "limite_superior": round(limite_sup, 2),
                "outliers": n,
                "% outliers": round(n / len(self.df) * 100, 2)
            })
        print(pd.DataFrame(resultados))

    def correlaciones_fuertes(self, umbral=0.5, variables=None):
        matriz = self.df[self._numericas(variables)].corr()
        columnas = matriz.columns
        encontradas = False
        for i in range(len(columnas)):
            for j in range(i + 1, len(columnas)):
                valor = matriz.iloc[i, j]
                if abs(valor) >= umbral:
                    print(f"{columnas[i]} y {columnas[j]}: {valor:.2f}")
                    encontradas = True
        if not encontradas:
            print(f"No hay correlaciones con valor absoluto mayor o igual a {umbral}")

    # ---------- GRAFICOS GENERICOS ----------

    # 1. DISTRIBUCION DE LAS VARIABLES NUMERICAS
    def graficar_distribucion_numericas(self, variables=None):
        for variable in self._numericas(variables):
            fig = px.histogram(
                self.df,
                x=variable,
                nbins=30,
                marginal="box",
                title=f"Distribución de {variable}",
                color_discrete_sequence=["#4169A1"]
            )
            fig.update_layout(template="plotly_white", hovermode="x unified")
            fig.show()

            serie = self.df[variable]
            print(
                f"Interpretación: Para {variable}, el promedio es {serie.mean():.2f} "
                f"y la mediana es {serie.median():.2f}. Los valores se encuentran entre "
                f"{serie.min()} y {serie.max()}. El gráfico permite identificar en qué "
                f"rangos se concentra la mayoría de los registros y observar posibles "
                f"valores extremos."
            )

    # 2. COMPARACION DE LAS VARIABLES NUMERICAS
    def graficar_comparacion_numericas(self, variables=None):
        variables = self._numericas(variables)
        datos_numericos = self.df[variables].melt(var_name="Variable", value_name="Valor")

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
            f"Interpretación: El gráfico permite comparar la variabilidad de "
            f"{', '.join(variables)}. Los puntos fuera de los bigotes representan "
            f"posibles valores atípicos. Cada variable tiene una escala diferente, "
            f"por lo que se debe interpretar considerando sus propias unidades."
        )

    # 3. FRECUENCIA DE LAS VARIABLES CATEGORICAS
    def graficar_frecuencia_categoricas(self, variables=None):
        for variable in self._categoricas(variables):
            frecuencias = self.df[variable].value_counts().reset_index()
            frecuencias.columns = [variable, "cantidad"]

            fig = px.bar(
                frecuencias,
                x=variable,
                y="cantidad",
                title=f"Cantidad de registros por {variable}",
                color="cantidad",
                color_continuous_scale="Blues"
            )
            fig.update_layout(template="plotly_white", xaxis_tickangle=-35)
            fig.show()

            categoria_mayor = frecuencias.iloc[0][variable]
            cantidad_mayor = int(frecuencias.iloc[0]["cantidad"])
            porcentaje = cantidad_mayor / len(self.df) * 100
            print(
                f"Interpretación: La categoría más frecuente en {variable} es "
                f"'{categoria_mayor}', con {cantidad_mayor} registros "
                f"({porcentaje:.1f}% del total)."
            )

    # 4. MATRIZ DE CORRELACION
    def graficar_matriz_correlacion(self, variables=None):
        matriz = self.df[self._numericas(variables)].corr()

        fig = px.imshow(
            matriz,
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
            "relaciones lineales más fuertes. La diagonal vale 1 porque cada variable "
            "tiene correlación perfecta consigo misma. Las correlaciones no representan "
            "necesariamente relaciones de causa y efecto."
        )