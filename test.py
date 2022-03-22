import cv2.cv2 as cv
imagePath = "./images/pant.jpeg"
img = cv.imread(imagePath)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img,(28,28))
img = 255.0 - img
img = img/255.0
print(img)
cv.imshow("h",img)
cv.waitKey()