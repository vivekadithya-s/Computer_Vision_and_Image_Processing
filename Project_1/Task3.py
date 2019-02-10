import cv2
import numpy as np
import os
cwd = os.getcwd()

# Please enter input image source path in next 2 lines
imageColour = cv2.imread(cwd+"/pos_10.jpg")
img = cv2.imread(cwd+"/pos_10.jpg",0)
img = cv2.GaussianBlur(img, (3,3),0.47)
laplacianimg = cv2.Laplacian(img,cv2.CV_8U)

# Please input template here
template = cv2.imread(cwd+"/template.jpg", 0)
laplaciantemp = cv2.Laplacian(template,cv2.CV_8U)

h, w = laplaciantemp.shape

res = cv2.matchTemplate(laplacianimg, laplaciantemp, cv2.TM_CCOEFF_NORMED)
threshold = 0.53

loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    x = cv2.rectangle(imageColour, pt, (pt[0]+w, pt[1]+h), (100, 10, 180), 2)
# Please enter destination path for result image
cv2.imshow('Detected', imageColour)
cv2.imwrite(cwd+"/CursorDetectedImage.png", imageColour)
print("\n\nPlease check for Detected Cursor window and close it when done :)")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Finished")
