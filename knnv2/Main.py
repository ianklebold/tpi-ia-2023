from knnv2 import ClassificatorService as classificatorService
from knnv2 import CalculateDistanceService as distanceService

print("Inicio de programa")
print("La clasificacion es : " + str(classificatorService
                                     .getClassification(6, distanceService.DistanceMethod.EUCLIDEAN)))
