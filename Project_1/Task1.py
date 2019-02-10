# Edge Detection using Sobel Operator
# By Vivek Adithya Srinivasa Raghavan
# UB Number 50290568
#import time
import cv2
#   numpy is used for np.zeros and np.array functions only
import numpy as np

#   Reading the source image into sample image matrix, please enter task1 source image path
sample = cv2.imread("task1.png",0)



#   Function to flip any image in both x and y axis.
def xyflip(image):
    #   Canvas for the flipped image using the source image as the basis
    img = image.copy()
    hhh = image.shape[0]
    www = image.shape[1]

    for i in range(hhh):
        for j in range(www):
            img[i][j] = image[hhh-i-1][www-j-1]
    #cv2.imshow("image copy", img)

    return img


#   Function to perform convolution on source image
#   based on supplied x or y sobel kernel
def convolution(image, kernel):
    #   Flips the kernel in both x and y axis before convolution
    kernel = xyflip(kernel)
    height = image.shape[0]
    width = image.shape[1]
    #   Kernel matrix height and width computation
    kheight = kernel.shape[0]
    kwidth = kernel.shape[1]

    h = kheight//2
    w = kwidth//2
    #   Initial canvas for the convolved image
    #   using the source image as the basis for size
    convolved_image = np.zeros(image.shape)

    for i in range(h, height-h):
        for j in range(w, width-w):
            sum = 0

            for m in range(kheight):
                for n in range(kwidth):
                    sum = (sum + kernel[m][n] * image[i-h+m][j-w+n])
            
            convolved_image[i][j] = sum

    return convolved_image


#   Function that normalizes two source images,
#   img_a with edges along x-axis and img_b with edges along y-axis
#   and normalizes the final image with edges in both axes
def normalization_func(img_a, img_b):
    final_img = np.zeros(img_a.shape)
    hh =img_a.shape[0]
    ww = img_a.shape[1]

    for i in range(hh):
        for j in range(ww):
            q = (img_a[i][j]**2 + img_b[i][j]**2)**(1/2)
            if(np.any(q > 90)):
                final_img[i][j] = 255
            else:
                final_img[i][j] = 0
    return final_img



#   Kernel for detecing edges in y-axis
kernel_y = np.zeros(shape=(3,3))
kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

#   grad_y is the final image with edges in y-axis
grad_y = convolution(sample, kernel_y)
cv2.imshow("Edges in Y-axis using Sobel", grad_y)
cv2.imwrite("YSobel.png", grad_y)



#   Kernel for detecing edges in x-axis
kernel_x = np.zeros(shape=(3,3))
kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

#   grad_x is the final image with edges in x-axis
grad_x = convolution(sample, kernel_x)
cv2.imshow("Edges in X-axis using Sobel", grad_x)
cv2.imwrite("XSobel.png", grad_x)



#   Final normalized image with all edges detected
#   in both x and y axes
final_sobel = normalization_func(grad_x, grad_y)
cv2.imshow("Final Sobel", final_sobel)
cv2.imwrite("Final_Sobel.png", final_sobel)


cv2.waitKey(0)
cv2.destroyAllWindows()

