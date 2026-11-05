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
        duplicados = self.__df.duplicated().sum()
        nulos_columna = self.__df.isnull().sum()

        nulos_tabla = pd.DataFrame({
            'Tipo de Dato': self.__df.dtypes,
            'Valores Nulos': nulos_columna,
            '% Nulos': (nulos_columna / total_filas) * 100
        })

        print("=== Reporte de Diagnóstico y Calidad ===")
        print(f"Total de registros: {total_filas}")
        print(f"Registros duplicados: {duplicados}")
        print("\nDetalle de campos y valores faltantes:")
        hay_nulos = duplicados > 0 or nulos_columna.sum() > 0
        print(nulos_tabla[nulos_tabla['Valores Nulos'] > 0] if hay_nulos else "No se detectaron valores nulos.")
        return nulos_tabla



    """Muestra los valores más comunes para cada columna categorica."""
    def resumen_categoricas(self, max_mostrar: int = 10):
        cols = self.__df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cols:
            print("No se encontraron columnas de texto.")
            return

        print(f"=== RESUMEN DE VARIABLES CATEGÓRICAS ({len(cols)} columnas) ===\n")
        for col in cols:
            total_unicos = self.__df[col].nunique()
            print("=" * 60)
            print(f"Columna: '{col}' | Total valores únicos: {total_unicos}")
            print("-" * 60)
            print(self.__df[col].value_counts(dropna=False).head(max_mostrar))
            if total_unicos > max_mostrar:
                print(f"... y {total_unicos - max_mostrar} valores más.")
            print()



    """Estandariza columnas numéricas continuas."""
    def estandarizar(self, columnas: list = None):
        if columnas is None:
            columnas = self.__df.select_dtypes(include=['number']).columns.tolist()

        if not columnas:
            print("No hay columnas numéricas para estandarizar.")
            return self.__df

        self.__df[columnas] = self.__scaler.fit_transform(self.__df[columnas])
        print(f"Columnas estandarizadas: {columnas}")
        return self.__df
