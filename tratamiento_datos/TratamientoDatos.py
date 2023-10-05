import pandas as pd

def tratarValoresNulos(df):
    # reemplaza los valores faltantes con el promedio de la columna correspondiente
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
    
    return df


def tratarValoresDeCategorias(df):
    # convierte la columna 'is_smoking' a tipo bool
    df['is_smoking'] = df['is_smoking'].apply(lambda x: 1 if x == 'YES' or x == 'NO' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'M' or x == 'F' else 0)
    
    # convierte las columnas 'education', 'prevalentStroke', 'prevalentHyp', 'diabetes', y 'TenYearCHD' a tipo int
    df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']] = df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']].astype(int)

    return df

def tratarNombresDeColumnas(df):
    # Renombrar las columnas
    df.rename(columns={
        # 'id': 'ID',
        'age': 'Edad',
        'education': 'Educación',
        'sex': 'Sexo',
        'is_smoking': 'Fumador',
        'cigsPerDay': 'Cigarrillos por Día',
        'BPMeds': 'Toma Medicamentos Pre ART',
        'prevalentStroke': 'Derrame Cerebral',
        'prevalentHyp': 'Hipertenso',
        'diabetes': 'Diabetes',
        'totChol': 'Colesterol Total',
        'sysBP': 'Presión Arterial Sistólica',
        'diaBP': 'Presión Arterial Diastólica',
        'BMI': 'IMC',
        'heartRate': 'Frecuencia Cardíaca',
        'glucose': 'Nivel de Glucosa',
        'TenYearCHD': 'Predicción'
    }, inplace=True)

    return df

def limpiar_datos(df):
    # df = pd.read_csv('resources\data_cardiovascular_risk.csv')

    df = tratarValoresNulos(df)
    df = tratarValoresDeCategorias(df)
    df = tratarNombresDeColumnas(df)
    # print(df.head())

    # pd.DataFrame.to_csv(df, 'data_tratada.csv', index=False)
    return df