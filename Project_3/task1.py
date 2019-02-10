import cv2
import numpy as np


def erosion(img):
    w = img.shape[0]
    h = img.shape[1]
    dil = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if (img[i][j] == 255):
                if (i == 0 or j == 0 or i == w - 1 or j == h - 1):
                    continue
            if (img[i - 1][j - 1] == 255 and img[i - 1][j] == 255 and img[i - 1][j + 1] == 255 and img[i][
                j - 1] == 255 and img[i][j] == 255 and img[i][j + 1] == 255 and img[i + 1][j - 1] == 255 and img[i + 1][
                j] == 255 and img[i + 1][j + 1] == 255):
                dil[i][j] = 255

    return dil


def dilation(img_r, r, c):
    for i in range(0, r - 2):
        for j in range(0, c - 2):
            if (img_r[i][j] == 255 or img_r[i][j + 1] == 255 or img_r[i][j + 2] == 255 or img_r[i + 1][j] == 255 or
                    img_r[i + 1][j + 2] == 255 or img_r[i + 2][j] == 255 or img_r[i + 2][j + 1] == 255 or img_r[i + 2][
                        j + 2] == 255):
                img_r[i][j] = 255
    return img_r


#Please change path
image_colour = cv2.imread('/Users/vivad/PycharmProjects/CLFinalProject/original_imgs/noise.jpg', 0)
image_dupe = image_colour.copy()
imageAA = image_colour.copy()
img_len = image_dupe.shape
r = img_len[0]
c = img_len[1]

# Morphology algorithm 1 -
# Closing followed by opening
# Closing Operation - Dilation + Erosion
dil1 = dilation(image_dupe, r, c)
dil_copy = dil1.copy()
after_erosion = erosion(dil_copy)

# Opening Operation - Erosion + Dilation
again_erosion = erosion(after_erosion)
erosion_copy = again_erosion.copy()
final_algo1_output = dilation(erosion_copy, r, c)

# Morphology algorithm 2
# Opening followed by closing
# Opening Operation - Erosion + Dilation
imageA = erosion(imageAA)
imageB = imageA.copy()
imageC = dilation(imageB, r, c)

# Closing Operation - Dilation + Erosion
imageD = dilation(imageC, r, c)
imageE = imageD.copy()
final_algo2_output = erosion(imageE)

#Please change path
cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task1/res_noise2.jpg", final_algo2_output)
cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task1/res_noise1.jpg", final_algo1_output)

#   Computing boundaries of images using difference between morphed image and its erosion result
Final_A = final_algo1_output.copy()
Final_B = final_algo2_output.copy()

erode_B = erosion(Final_B)
erode_A = erosion(Final_A)
res_bound1 = Final_A - erode_A
res_bound2 = Final_B - erode_B


#Please change path
cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task1/res_bound1.jpg", res_bound1)
cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task1/res_bound2.jpg", res_bound2)


