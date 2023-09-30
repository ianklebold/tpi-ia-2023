#Dataset va a almacenar el conjunto de instancias que se van a clasificar.
class Dataset:
    def __init__(self, data=None):
        if data is None:
            data = list()
        self.data = data
