# Mandar esto a una carpeta a parte
# Un conjunto es una lista de instancias con la misma clasificacion
from knnv2 import CalculateDistanceService


class Conjunto:
    def __init__(self, clasificacion=None, lista_de_instancias=None):
        if lista_de_instancias is None:
            lista_de_instancias = list()
        self.lista_de_instancias = lista_de_instancias  # Lista de instancias con la clasificacion
        self.classification = clasificacion  # Clase del conjunto


class Clase:
    def __init__(self, identificador):
        self.nombre = identificador


n = 2  # Cantidad de conjuntos de clases


def calculateClassification(vecinos_mas_cercanos, classification):
    # d(c1,f(x1))+  d(c1,f(x2)) +  d(c1,f(x3))
    amount = 0
    for neighbor in vecinos_mas_cercanos:
        if neighbor[0].clase == classification:
            amount += 1
        else:
            amount += 0
    return amount


def getClasses(list_of_identifications):
    classes = list()
    for ident in list_of_identifications:
        classes.append(Clase(ident))

    return classes


def getSets(list_of_classes, vecinos_mas_cercanos):
    sets = list()
    for class_instance in list_of_classes:
        list_of_classes.append(Conjunto(class_instance.nombre, vecinos_mas_cercanos))
    return sets


def getClassification():
    k_neighbors = 6
    classes = getClasses([0, 1])
    neighbors_more_earn = CalculateDistanceService.findNeighborMoreEarn(k_neighbors)
    sets = getSets(classes, neighbors_more_earn)
    sets_classified = list()

    for set_instance in sets:
        amount = calculateClassification(set_instance.lista_de_instancias, set_instance.classification)
        sets_classified.append((set_instance, amount))

    sets_classified.sort(key=lambda x: x[1])

    return sets_classified.pop(0)
