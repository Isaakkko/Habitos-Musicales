# LIBRERIAS
import pandas as pd
from sklearn.preprocessing import StandardScaler


class UMAP:

    def __init__(self, datos):
        # acepta un dataframe o la ruta del csv limpio
        if isinstance(datos, pd.DataFrame):
            self.df = datos.copy()
        else:
            self.df = pd.read_csv(datos)
        self.variables = None
        self.datos_est = None

    # selecciona las variables y las estandariza (media 0, desviacion 1)
    def preparar_datos(self, variables):
        self.variables = variables
        seleccion = self.df[variables].astype(float)
        self.datos_est = pd.DataFrame(
            StandardScaler().fit_transform(seleccion),
            columns=variables,
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