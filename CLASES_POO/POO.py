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




































