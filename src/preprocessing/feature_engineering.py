# Ajuste de columnas para el DataFrame de vecinos por k
columnas = ['K'] + [f'i{i}' for i in range(1, 7)]
# Añadimos una lista vacía para ajustar las columnas
vecinos_por_k.append([''] * len(columnas))
coincidencias_por_k.append([''] * len(columnas))

# Crear el DataFrame para mostrar los vecinos por cada k
df_vecinos_por_k = pd.DataFrame(vecinos_por_k, columns=columnas)
# Eliminar la última fila del DataFrame df_vecinos_por_k
df_vecinos_por_k = df_vecinos_por_k.drop(df_vecinos_por_k.index[-1])

# Crear el DataFrame para mostrar las coincidencias por cada k
df_coincidencias_por_k = pd.DataFrame(coincidencias_por_k, columns=columnas)
# Eliminar la última fila del DataFrame df_coincidencias_por_k
df_coincidencias_por_k = df_coincidencias_por_k.drop(df_coincidencias_por_k.index[-1])

# Transponer solo el contenido debajo de las columnas de las ix
df_vecinos_por_k.iloc[:, 1:] = df_vecinos_por_k.iloc[:, 1:].values.T
df_coincidencias_por_k.iloc[:, 1:] = df_coincidencias_por_k.iloc[:, 1:].values.T

# Imprimir el DataFrame con los resultados de vecinos por k
print("\nVecinos más cercanos por cada k para cada instancia:")
print(df_vecinos_por_k)

# Imprimir el DataFrame con los resultados de coincidencias por k
print("\nCoincidencias por cada k para cada instancia:")
print(df_coincidencias_por_k)
