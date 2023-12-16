import cv2
import matplotlib.pyplot as plt

img_path = 'test.jpeg'
img = cv2.imread(img_path)

imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imagen_gris = cv2.resize(imagen_gris, (416,416), interpolation=cv2.INTER_LINEAR)

borders_canny = cv2.Canny(img, 200,220)

print(borders_canny.shape)

cv2.imshow('Imagen en Escala de Grises', imagen_gris)
cv2.waitKey(0)
cv2.destroyAllWindows()