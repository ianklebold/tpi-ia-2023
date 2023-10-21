from math import sqrt

def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += pow(instance1[i] - instance2[i], 2)
    return sqrt(distance)

def get_neighbors(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        dist = euclidean_distance(train_instance[:-1], test_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train_set, test_instance, k):
    neighbors = get_neighbors(train_set, test_instance, k)
    class_votes = {}
    for neighbor in neighbors:
        class_name = neighbor[-1]
        if class_name in class_votes:
            class_votes[class_name] += 1
        else:
            class_votes[class_name] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]


# Clasificación utilizando distancia euclidiana
test_instance = (2, 2)
train_set = {(1, 2, 'a'), (2, 3, 'b'), (3, 1, 'a'), (4, 2, 'b')}


k = 3
predicted_class = predict_classification(train_set, test_instance, k)
print('La instancia de prueba euclediana pertenece a la clase:', predicted_class)
