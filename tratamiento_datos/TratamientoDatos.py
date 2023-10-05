
def tratarValoresNulos(df):
    # reemplaza los valores faltantes con el promedio de la columna correspondiente
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

    # convierte las columnas 'education', 'prevalentStroke', 'prevalentHyp', 'diabetes', y 'TenYearCHD' a tipo int
    df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']] = df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']].astype(int)

def tratarValoresDeCategorias(df):
    return

def tratarNombresDeColumnas(data):
    # Extraer las columnas que deseas graficar
    id = data['id']
    age = data['age']
    education = data['education']
    sex = data['sex']
    is_smoking = data['is_smoking']
    cigsPerDay = data['cigsPerDay']
    BPMeds = data['BPMeds']
    prevalentStroke = data['prevalentStroke']
    prevalentHyp = data['prevalentHyp']
    diabetes = data['diabetes']
    totChol = data['totChol']
    sysBP = data['sysBP']
    diaBP = data['diaBP']
    BMI = data['BMI']
    heartRate = data['heartRate']
    glucose = data['glucose']
    TenYearCHD = data['TenYearCHD']

    return data

