import pandas as pd
from sklearn.preprocessing import StandardScaler

class about_data:
    def __init__(self, origen):
        # En caso de pasar la ruta del csv
        if isinstance(origen, str):
            self.__ruta = origen
            self.__df = pd.read_csv(origen)
            print("Datos cargados con éxito desde la ruta")
        # En caso de pasar un DataFrame
        elif isinstance(origen, pd.DataFrame):
            self.__ruta = None
            self.__df = origen.copy()
            print("Datos cargados con éxito a partir de la tabla")
        else:
            raise TypeError("El parámetro debe ser una ruta (str) o un DataFrame de pandas.")
        self.__scaler = StandardScaler()

    @property
    def datos(self):
        return self.__df # Devuelve el DataFrame cargado para manipularlo.



    """Muestra filas, duplicados y columnas con nulos."""
    def reporte_calidad(self):
        total_filas = len(self.__df)
        duplicados = self.__df.duplicated().sum() # Función que mostrara las columnas que tengan duplicados
        nulos_columna = self.__df.isnull().sum() # Función que mostrara las columnas que tengan nulls

        #Creación de un DataFrame donde se muestra el tipo de dato, Valores nulos y su porcentaje
        nulos_tabla = pd.DataFrame({
            'Tipo de Dato': self.__df.dtypes,
            'Valores Nulos': nulos_columna,
            '% Nulos': (nulos_columna / total_filas) * 100
        })

        print("=== Reporte de Diagnóstico y Calidad ===")
        print(f"Total de registros: {total_filas}")
        print(f"Registros duplicados: {duplicados}")
        print("\nDetalle de campos y valores faltantes:")
        hay_nulos = duplicados > 0 or nulos_columna.sum() > 0 #En caso de que se encuentren duplicados
        print(nulos_tabla[nulos_tabla['Valores Nulos'] > 0] if hay_nulos else "No se detectaron valores nulos.")
        return nulos_tabla



    """Muestra los valores más comunes para cada columna categorica."""
    def resumen_categoricas(self, max_mostrar: int = 10): #max_mostrar inicialmente esta 10 aumentar en caso de que sea necesario (columna tiene muchas categorias)
        cols = self.__df.select_dtypes(include=['object', 'category']).columns.tolist() #tolist()) extrae los nombres de las columnas que sean de texto o categorias
        if not cols: # Aviso que indica que no se encontaron columnas de texto en caso del que el df no tenga
            print("No se encontraron columnas de texto.")
            return

        print(f"=== RESUMEN DE VARIABLES CATEGÓRICAS ({len(cols)} columnas) ===\n")
        for col in cols:
            total_unicos = self.__df[col].nunique() # Busqueda de registros unicos en la columna categorica
            print("=" * 60)
            print(f"Columna: '{col}' | Total valores únicos: {total_unicos}")
            print("-" * 60)
            print(self.__df[col].value_counts(dropna=False).head(max_mostrar))
            if total_unicos > max_mostrar:
                print(f"... y {total_unicos - max_mostrar} valores más.")
            print()



    """Estandariza columnas numéricas continuas."""
    def estandarizar(self, columnas: list = None): #None selecciona todas las columnas numéricas automáticamente en caso de no ser seleccionadas.
        if columnas is None:
            columnas = self.__df.select_dtypes(include=['number']).columns.tolist() #Se ejecuta solo si es numerica

        if not columnas:
            print("No hay columnas numéricas para estandarizar.")
            return self.__df

        self.__df[columnas] = self.__scaler.fit_transform(self.__df[columnas]) #fit_transform estandariza los datos de la columna numerica
        print(f"Columnas estandarizadas: {columnas}")
        return self.__df


#*************************************REGLAS DE ASOCIACIÓN************************************************************
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

#Clase que tiene el proposito de implementar las reglas de asociación con la faciliadad de cambiar paremtros
class Reglas_Asociacion:
    def __init__(self, df: pd.DataFrame):
        # Aseguramos que la matriz sea 100% de booleanos (True/False)
        self.df = df.astype(bool)
        self.items_frecuentes = None # None = Contenedores vacios temporales
        self.reglas = None # None = Contenedores vacios temporales
        self.metodo = None # None = Contenedores vacios temporales

    """
    Entrena el algoritmo seleccionado: 'apriori', 'fpgrowth' o 'eclat'.
    """
    def fit(self, metodo: str = 'fpgrowth', min_support: float = 0.03, min_confidence: float = 0.4):
        self.metodo = metodo.lower()

        # 1. Buscar conjuntos frecuentes
        # ECLAT encuentra los mismos itemsets frecuentes que FP-Growth (minería vertical)
        if self.metodo == 'apriori':
            self.items_frecuentes = apriori(self.df, min_support=min_support, use_colnames=True) #De la liberia ejecuta la función apriori ejecutando este algoritmo y toma el soporte configurado
        elif self.metodo in ['fpgrowth', 'eclat']: #De la liberia ejecuta la función fpgrowth ejecutando este algoritmo y recordar que ECLAT encuentra los mismos itemsets que fpgrowth y toma el soporte configurado
            self.items_frecuentes = fpgrowth(self.df, min_support=min_support, use_colnames=True)
        else:
            raise ValueError("Método no válido. Usa 'apriori', 'fpgrowth' o 'eclat'.") # En caso de que el parametro colocado no sea el correcto

        # 2. Generar las reglas de asociación
        if self.items_frecuentes is not None and not self.items_frecuentes.empty:
            self.reglas = association_rules( # association_rules toma los ítems frecuentes que superaron el soporte y calcula únicamente las reglas cuya certeza (confianza) sea igual o mayor al umbral fijado.
                self.items_frecuentes,
                metric='confidence',
                min_threshold=min_confidence
            )
        else:
            print(f"No se encontraron combinaciones con soporte >= {min_support}")
            self.reglas = pd.DataFrame()

        return self.reglas

    def ver_reglas(self, top_n: int = 10, ordenar_por: str = 'lift', lift_minimo: float = 1.0):
        """Devuelve un DataFrame limpio ordenado por la métrica elegida."""
        if self.reglas is None or self.reglas.empty:
            print("No hay reglas disponibles.")
            return pd.DataFrame()

        # Filtramos para descartar reglas irrelevantes
        df_filtrado = self.reglas[self.reglas['lift'] >= lift_minimo].copy()
        columnas = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        return df_filtrado.sort_values(by=ordenar_por, ascending=False).head(top_n)[columnas]

    def graficar_dispersion(self):
        """Gráfico de dispersión simple: Soporte vs Confianza (con color de Lift)."""
        if self.reglas is None or self.reglas.empty:
            print("No hay reglas para graficar.")
            return

        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(
            self.reglas['support'],
            self.reglas['confidence'],
            c=self.reglas['lift'],
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Soporte (Support)')
        plt.ylabel('Confianza (Confidence)')
        plt.title(f'Reglas de Asociación ({self.metodo.upper()})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def graficar_red(self, top_n: int = 10):
        """Grafo dirigido mostrando las conexiones de las reglas más fuertes."""
        if self.reglas is None or self.reglas.empty:
            print("No hay reglas para graficar.")
            return

        # Graficamos solo el top_n para evitar saturar el gráfico
        top_reglas = self.reglas.sort_values(by='lift', ascending=False).head(top_n)

        G = nx.DiGraph()
        for _, row in top_reglas.iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            G.add_edge(ant, con, weight=row['lift'])

        plt.figure(figsize=(9, 6))
        pos = nx.spring_layout(G, k=0.8)
        nx.draw(
            G, pos, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=2500, font_size=8, arrows=True
        )
        plt.title(f"Red de Reglas Principales ({self.metodo.upper()})")
        plt.show()


#**************************************Modelado (Clusterización)**************************
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class Reduccion_Dimensionalidad:
    def __init__(self, df_escalado: pd.DataFrame, df_original: pd.DataFrame):
        """
        df_escalado: DataFrame con columnas numéricas ya estandarizadas (StandardScaler).
        df_original: DataFrame original para colorear los puntos por variables categóricas (ej. subscription, top_genre).
        """
        self.X = df_escalado.copy()
        self.df_base = df_original.copy()
        self.pca_df = None
        self.tsne_df = None
        self.umap_df = None
        self.varianza_pca = None

    def fit_pca(self, n_components: int = 2):
        """Aplica PCA lineal y guarda la varianza explicada."""
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(self.X)
        self.varianza_pca = pca.explained_variance_ratio_

        cols = [f'PC{i + 1}' for i in range(n_components)]
        self.pca_df = pd.DataFrame(coords, columns=cols, index=self.df_base.index)

        total_var = sum(self.varianza_pca) * 100
        print(f"PCA entrenado | Varianza explicada total ({n_components} comp.): {total_var:.2f}%")
        for i, v in enumerate(self.varianza_pca):
            print(f"  - PC{i + 1}: {v * 100:.2f}%")
        return self.pca_df

    def fit_tsne(self, perplexity: float = 30.0):
        """Aplica t-SNE (proyección no lineal enfocada en estructura local)."""
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        coords = tsne.fit_transform(self.X)
        self.tsne_df = pd.DataFrame(coords, columns=['TSNE_1', 'TSNE_2'], index=self.df_base.index)
        print(f"t-SNE entrenado (perplexity={perplexity}).")
        return self.tsne_df

    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1):
        """Aplica UMAP (proyección no lineal que preserva estructura local y global)."""
        reductor = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        coords = reductor.fit_transform(self.X)
        self.umap_df = pd.DataFrame(coords, columns=['UMAP_1', 'UMAP_2'], index=self.df_base.index)
        print(f"UMAP entrenado (n_neighbors={n_neighbors}, min_dist={min_dist}).")
        return self.umap_df

    def graficar(self, metodo: str = 'pca', color_col: str = 'subscription'):
        """
        Grafica la proyección 2D elegida ('pca', 'tsne' o 'umap') coloreada por una variable.
        """
        metodo = metodo.lower()
        if metodo == 'pca' and self.pca_df is not None:
            df_plot = self.pca_df.copy()
            x_col, y_col = 'PC1', 'PC2'
            titulo = f"Proyección PCA (Varianza: {sum(self.varianza_pca[:2]) * 100:.1f}%)"
        elif metodo == 'tsne' and self.tsne_df is not None:
            df_plot = self.tsne_df.copy()
            x_col, y_col = 'TSNE_1', 'TSNE_2'
            titulo = "Proyección t-SNE (Estructura Local)"
        elif metodo == 'umap' and self.umap_df is not None:
            df_plot = self.umap_df.copy()
            x_col, y_col = 'UMAP_1', 'UMAP_2'
            titulo = "Proyección UMAP (Estructura Local y Global)"
        else:
            raise ValueError("Método no válido o no entrenado aún. Ejecuta fit_pca(), fit_tsne() o fit_umap() primero.")

        df_plot[color_col] = self.df_base[color_col].astype(str)
        fig = px.scatter(
            df_plot, x=x_col, y=y_col, color=color_col,
            title=f"{titulo} — Agrupado por '{color_col}'",
            opacity=0.75
        )
        fig.show()



from math import ceil, floor, pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sb
from seaborn import color_palette
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _columnas_numericas_acp(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["number"]).columns.tolist()


def ACP_clustering(datos_est: pd.DataFrame, n_clusters: int = 3, max_iter: int = 2000, n_init: int = 150, random_state: int = 42) -> pd.DataFrame:
    acp = PCA(n_components=2, random_state=random_state)
    df_acp = pd.DataFrame(acp.fit_transform(datos_est), columns=["dim1", "dim2"], index=datos_est.index)

    kmedias = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    df_acp["cluster"] = kmedias.fit_predict(df_acp)
    return df_acp


class GraficosClustering:
    def __init__(self, datos_original: pd.DataFrame, df_acp: pd.DataFrame):
        self.datos = datos_original
        self.datos_num = datos_original[_columnas_numericas_acp(datos_original)]
        self.df_acp = df_acp
        self.grupos = df_acp["cluster"]
        self.centros = None

    def scatter_2D(self, label: str = "cluster", title: str = "Clústeres en ACP"):
        fig = px.scatter(
            self.df_acp,
            x="dim1",
            y="dim2",
            color=self.grupos.astype(str),
            hover_data={"Individuo": self.df_acp.index.values},
            labels={"color": label},
        )
        fig.update_layout(title_text=title, title_x=0.5)
        fig.show()

    def calcular_centros(self, cant_clusters: int | None = None) -> np.ndarray:
        if cant_clusters is None:
            cant_clusters = int(self.grupos.nunique())

        filas = [self.datos_num[self.grupos == k].mean() for k in range(cant_clusters)]
        self.centros = np.array(pd.DataFrame(filas))
        return self.centros

    def bar_plot(self, scale: bool = False, cluster: list | None = None, var: list | None = None):
        if self.centros is None:
            self.calcular_centros()

        centros = np.copy(self.centros)
        labels = np.array(self.datos_num.columns)

        if scale:
            for col in range(centros.shape[1]):
                max_val = np.max(centros[:, col])
                if max_val != 0:
                    centros[:, col] /= max_val

        if var is not None:
            var_set = set(var)
            mask = np.array([x in var_set for x in labels])
            centros = centros[:, mask]
            labels = labels[mask]

        clusters_a_mostrar = list(range(centros.shape[0])) if cluster is None else list(cluster)
        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        maximo = ceil(centros.max())

        for pos, idx_cluster in enumerate(clusters_a_mostrar, start=1):
            plt.subplot(1, len(clusters_a_mostrar), pos)
            valores = centros[idx_cluster].tolist()
            plt.barh(range(len(valores)), valores, 1 / 1.5, color=colores[: len(valores)])
            plt.xlim(minimo, maximo)
            plt.title(f"Cluster {idx_cluster}")
            if pos == 1:
                plt.yticks(range(len(labels)), labels)
            else:
                plt.yticks([])

        plt.tight_layout()
        plt.show()

    def bar_plot_detail(self, columns_to_plot: list | None = None, figsize: tuple = (10, 7), dpi: int = 150):
        if self.centros is None:
            self.calcular_centros()

        columns_names = list(self.datos_num.columns)
        df_centros = pd.DataFrame(
            self.centros,
            columns=columns_names,
            index=[f"Cluster {i}" for i in range(self.centros.shape[0])],
        )

        if not columns_to_plot:
            cols = columns_names
        elif isinstance(columns_to_plot[0], str):
            cols = columns_to_plot
        else:
            cols = [columns_names[i] for i in columns_to_plot]

        plots = len(cols)
        rows, total_cols = ceil(plots / 2), 2

        plt.figure(1, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(hspace=1, wspace=0.5)

        for idx, col_name in enumerate(cols):
            r, c = divmod(idx, total_cols)
            ax = plt.subplot2grid((rows, total_cols), (r, c), colspan=1, rowspan=1)
            sb.barplot(y=df_centros.index, x=col_name, data=df_centros, ax=ax)

        plt.show()

    def radar_plot(self):
        if self.centros is None:
            self.calcular_centros()

        labels = list(self.datos_num.columns)
        centros_norm = np.array([
            ((n - np.min(n)) / (np.max(n) - np.min(n)) * 100)
            if np.max(n) != np.min(n)
            else np.full_like(n, 50.0, dtype=float)
            for n in self.centros.T
        ])

        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]

        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
            color="grey",
            size=8,
        )
        plt.ylim(-10, 100)

        for i in range(centros_norm.shape[1]):
            valores = centros_norm[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1, linestyle="solid", label=f"Cluster {i}")
            ax.fill(angulos, valores, alpha=0.3)

        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.show()





























