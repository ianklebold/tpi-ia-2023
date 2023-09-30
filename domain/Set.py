class Set:
    def __init__(self, classification=None, listofinstances=None):
        if listofinstances is None:
            listofinstances = list()
        self.listofinstances = listofinstances  # Lista de instancias con la clasificacion
        self.classification = classification  # Clase del conjunto
