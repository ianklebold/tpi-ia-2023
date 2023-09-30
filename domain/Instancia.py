# Una instancia va a ser cada registro que tenemos en el CSV que vamos a clasificar.
class Instancia:
    def __init__(self, x, y, clasificacion=None):
        self.x = x
        self.y = y
        self.clasificacion = clasificacion

