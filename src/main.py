import random
import pandas as pd
from preprocessing.data_preparation import DataPreparation
import preprocessing.data_preparation
import knn

def random_gen():
    dataset = DataPreparation.get_dataset('resources/dataset_cleaned_fixed.csv')
    renamed_dataset = DataPreparation.rename_dataset(dataset)
    instances = DataPreparation.get_ds_instances(renamed_dataset)

    random_instances = random.sample(instances, 6)

    random_results = []

    results = []

    k_neighbors = []

    k_coincidences = []

    for i, new_instance in enumerate(random_instances, start=1):
        # Lista para almacenar los resultados para esta instancia aleatoria
        instance_results = []

        # Lista para almacenar los vecinos más cercanos para cada k
        neighbors_current_k = []
        k_current_coincidences = []

        # Iterar para k de 1 a 5
        for k in range(1, 7):
            # Encuentra los vecinos más cercanos
            nearest_neighbors = knn.training_algorithm(instances, new_instance, k)

            # Calcula la distancia para cada vecino
            for neighbor in nearest_neighbors:
                neighbor.distance = knn.calculate_distance(neighbor, new_instance)

            # Ordena los vecinos por distancia
            nearest_neighbors.sort(key=lambda x: x.distance)

            # Crea un DataFrame para mostrar los resultados
            preprocessing.data_preparation.print_neighbors(nearest_neighbors, k)

            # Almacena el ID del último vecino más cercano para esta instancia y k
            neighbors_current_k.append(nearest_neighbors[-1].id)

            # Almacena la cantidad de coincidencias
            k_current_coincidences.append(
              sum(1 for neighbor in nearest_neighbors if neighbor.get_class() == new_instance.get_class()))

            # Cuenta cuántas predicciones coinciden con el valor de la clase
            coincidences = sum(1 for neighbor in nearest_neighbors if neighbor.getClass() == new_instance.get_class())

            # Almacena los resultados
            instance_results.append({'K': f'k={k}', 'Coincidences': coincidences})

        k_neighbors.append([f'k={i}'] + neighbors_current_k)
        k_coincidences.append([f'k={i}'] + k_current_coincidences)

        # Crea un DataFrame para mostrar los resultados de coincidencias con la clase para esta instancia
        df_instance_coincidences = pd.DataFrame(instance_results)
        print(f"\nResults for instance: i{i}:")
        print(df_instance_coincidences)

