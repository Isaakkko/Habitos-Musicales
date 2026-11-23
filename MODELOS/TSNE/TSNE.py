"""Análisis t-SNE independiente para hábitos de streaming musical.

No requiere ACP, clustering ni clases de otros integrantes.
"""
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class AnalisisTSNE:
    """Calcula t-SNE y presenta cinco vistas del mismo resultado."""

    def __init__(self, datos: pd.DataFrame, columnas_modelo=None):
        self.datos = datos.copy()
        self.columnas_modelo = columnas_modelo if columnas_modelo is not None else [
            'age', 'daily_listening_minutes', 'playlists_count', 'skip_rate_pct'
        ]
        faltantes = [c for c in self.columnas_modelo if c not in self.datos.columns]
        if faltantes:
            raise ValueError(f'Faltan columnas numéricas: {faltantes}')
        self.X_escalado = None
        self.tsne_df = None

    def ajustar(self, perplexity=30, random_state=42):
        """Estandariza las variables y calcula una única proyección t-SNE."""
        numericos = self.datos[self.columnas_modelo].apply(pd.to_numeric, errors='coerce')
        if numericos.isna().any().any():
            raise ValueError('Hay valores nulos o no numéricos en las variables del t-SNE.')
        if len(numericos) < 3 or not (0 < perplexity < len(numericos)):
            raise ValueError('perplexity debe ser mayor que 0 y menor que el número de filas (mínimo 3).')
        self.X_escalado = pd.DataFrame(
            StandardScaler().fit_transform(numericos),
            columns=self.columnas_modelo, index=self.datos.index
        )
        coordenadas = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state,
            init='pca', learning_rate='auto'
        ).fit_transform(self.X_escalado)
        self.tsne_df = pd.DataFrame(
            coordenadas, columns=['TSNE_1', 'TSNE_2'], index=self.datos.index
        )
        return self.tsne_df

    def graficar(self, color_col, titulo=None):
        """Colorea el t-SNE según una categoría existente en el CSV."""
        if self.tsne_df is None:
            raise ValueError('Primero ejecutá ajustar().')
        if color_col not in self.datos.columns:
            raise ValueError(f'No existe la columna {color_col!r}.')
        tabla = self.tsne_df.copy()
        tabla[color_col] = self.datos.loc[tabla.index, color_col].fillna('Sin dato').astype(str)
        return self._grafico(tabla, color_col, titulo or f't-SNE | {color_col}')

    def graficar_intensidad_escucha(self):
        """Agrupa descriptivamente los minutos diarios en terciles, sin clustering."""
        if self.tsne_df is None:
            raise ValueError('Primero ejecutá ajustar().')
        minutos = pd.to_numeric(self.datos.loc[self.tsne_df.index, 'daily_listening_minutes'], errors='coerce')
        if minutos.isna().any():
            raise ValueError('daily_listening_minutes contiene valores no válidos.')
        # Rank permite crear tres niveles aun cuando haya valores repetidos.
        niveles = pd.qcut(
            minutos.rank(method='first'), q=3,
            labels=['Baja', 'Media', 'Alta']
        )
        tabla = self.tsne_df.copy()
        tabla['intensidad_escucha'] = niveles.astype(str)
        return self._grafico(
            tabla, 'intensidad_escucha',
            't-SNE | Intensidad de escucha diaria',
            colores={'Baja': '#636EFA', 'Media': '#00CC96', 'Alta': '#EF553B'}
        )

    @staticmethod
    def _grafico(tabla, columna, titulo, colores=None):
        opciones = dict(color_discrete_map=colores) if colores else dict(
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig = px.scatter(
            tabla, x='TSNE_1', y='TSNE_2', color=columna,
            title=titulo, opacity=0.78, template='plotly_dark',
            labels={'TSNE_1': 'Dimensión t-SNE 1', 'TSNE_2': 'Dimensión t-SNE 2'},
            **opciones
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_layout(
            height=600, autosize=True, title_x=0.5,
            paper_bgcolor='#101222', plot_bgcolor='#101222',
            font=dict(size=13, color='#F5F3FF'),
            legend_title_text=columna,
            margin=dict(l=55, r=35, t=85, b=60)
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.09)', zeroline=False)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.09)', zeroline=False)
        return fig
