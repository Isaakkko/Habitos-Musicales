# LIBRERIAS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px


class MODELO_UMAP:

    def __init__(self, datos):
        # acepta un dataframe o la ruta del csv limpio
        if isinstance(datos, pd.DataFrame):
            self.df = datos.copy()
        else:
            self.df = pd.read_csv(datos)
        self.variables = None
        self.datos_est = None
        self.df_umap = None

    # selecciona las variables, codifica las categoricas (one-hot) y estandariza
    def preparar_datos(self, variables, categoricas=None):
        self.variables = variables
        seleccion = self.df[variables].astype(float)

        if categoricas:
            dummies = pd.get_dummies(self.df[categoricas], dtype=float)
            seleccion = pd.concat([seleccion, dummies], axis=1)

        self.datos_est = pd.DataFrame(
            StandardScaler().fit_transform(seleccion),
            columns=seleccion.columns,
            index=self.df.index
        )
        print(self.datos_est.head())
        print(self.datos_est.shape)

    # reduce las variables estandarizadas a 2 dimensiones con UMAP
    def aplicar_umap(self, n_neighbors=15, random_state=42):
        umap_est = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state
        ).fit_transform(self.datos_est)

        self.df_umap = pd.DataFrame(
            umap_est,
            columns=["dim1", "dim2"],
            index=self.datos_est.index
        )
        print(self.df_umap.head())

    # agrupa con KMeans sobre las 2 dimensiones de UMAP
    def agrupar(self, n_clusters=3, random_state=42):
        kmedias = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        self.df_umap["cluster"] = kmedias.fit_predict(self.df_umap[["dim1", "dim2"]])
        print(self.df_umap["cluster"].value_counts().sort_index())

    # grafica las 2 dimensiones de UMAP coloreadas por cluster
    def graficar_clusters(self, hover=None):
        datos = self.df_umap.copy()
        datos["cluster"] = datos["cluster"].astype(str)

        # columnas del dataset original que se muestran al pasar el cursor
        if hover:
            datos = datos.join(self.df[hover])

        fig = px.scatter(
            datos,
            x="dim1",
            y="dim2",
            color="cluster",
            hover_data=hover,
            title="Clusters sobre las dimensiones de UMAP"
        )
        fig.update_layout(template="plotly_white")
        fig.show()

    # promedio de cada variable por cluster, en la escala original
    def perfil_clusters(self):
        datos = self.df[self.variables].astype(float)
        datos["cluster"] = self.df_umap["cluster"]

        perfil = datos.groupby("cluster").mean().round(2)
        perfil["cantidad"] = datos["cluster"].value_counts().sort_index()
        print(perfil)

    # promedio de cada variable estandarizada por cluster
    def _perfil_estandarizado(self):
        datos = self.datos_est.copy()
        datos["cluster"] = self.df_umap["cluster"]
        return datos.groupby("cluster").mean()

    # barras con el promedio estandarizado de cada variable en cada cluster
    def graficar_barras_clusters(self):
        perfil = self._perfil_estandarizado().reset_index()
        largo = perfil.melt(id_vars="cluster", var_name="variable", value_name="promedio")
        largo["cluster"] = largo["cluster"].astype(str)

        fig = px.bar(
            largo,
            x="promedio",
            y="variable",
            color="variable",
            facet_col="cluster",
            orientation="h",
            title="Perfil de cada cluster (promedio estandarizado)"
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("cluster=", "Cluster ")))
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.show()

    # radar para comparar la forma de cada cluster
    def graficar_radar_clusters(self):
        perfil = self._perfil_estandarizado()

        # se escala cada variable de 0 a 100 entre clusters (0 = el cluster mas bajo, 100 = el mas alto)
        minimo = perfil.min()
        rango = (perfil.max() - minimo).replace(0, float("nan"))
        escalado = ((perfil - minimo) / rango * 100).fillna(50)

        largo = escalado.reset_index().melt(id_vars="cluster", var_name="variable", value_name="valor")
        largo["cluster"] = largo["cluster"].astype(str)

        fig = px.line_polar(
            largo,
            r="valor",
            theta="variable",
            color="cluster",
            line_close=True,
            range_r=[0, 100],
            title="Radar de los clusters"
        )
        fig.update_traces(fill="toself", opacity=0.5)
        fig.update_layout(template="plotly_white")
        fig.show()