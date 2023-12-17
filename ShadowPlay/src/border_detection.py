import cv2
import matplotlib.pyplot as plt

img_path = 'test.jpeg'
img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (416,416), interpolation=cv2.INTER_LINEAR)

img = cv2.Canny(img,100,220)

print(img.shape)

cv2.imshow('Imagen en Escala de Grises', img)
cv2.waitKey(0)
cv2.destroyAllWindows()