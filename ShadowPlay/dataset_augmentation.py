import albumentations as A
import cv2

# Transform
transform = A.Compose([
    #A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=1),
])

image = cv2.imread('data/apple.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(type(image))

transformed = transform(image=image)
transformed_image = transformed['image']
transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

print(type(transformed_image))

cv2.imshow('image test',transformed_image)
cv2.imwrite('data/apple_3.jpg',transformed_image)
#cv2.waitKey(0)

