# Un conjunto es una lista de instancias con la misma clasificacion
class Conjunto:
    def __init__(self, clasificacion=None, lista_de_instancias=None):
        if lista_de_instancias is None:
            lista_de_instancias = list()
        self.lista_de_instancias = lista_de_instancias  # Lista de instancias con la clasificacion
        self.classification = clasificacion  # Clase del conjunto

