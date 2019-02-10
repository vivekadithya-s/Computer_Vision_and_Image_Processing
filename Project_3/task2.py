import cv2
import numpy as np
import skimage.color
import matplotlib.pyplot as plt

image_colour = cv2.imread("/Users/vivad/PycharmProjects/CLFinalProject/original_imgs/turbine-blade.jpg", 1)
image_gray = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

ker = np.ndarray((3, 3), dtype='int')
ker[0][0] = 0
ker[0][1] = 1
ker[0][2] = 0
ker[1][0] = 1
ker[1][1] = -4
ker[1][2] = 1
ker[2][0] = 0
ker[2][1] = 1
ker[2][2] = 0


def function_convolution(source, masking_kernel):
    image_h = source.shape[0]
    image_w = source.shape[1]

    kernel_h = masking_kernel.shape[0]
    kernel_w = masking_kernel.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    convolved_image = np.zeros(source.shape)

    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = (sum + masking_kernel[m][n] * source[i - h + m][j - w + n])

            if (sum > 310):
                convolved_image[i][j] = 255

    return convolved_image





conv_image_y_axis = function_convolution(image_gray, ker)
final_op_image = skimage.color.gray2rgb(conv_image_y_axis)
final_labelled = skimage.color.gray2rgb(conv_image_y_axis)
for i in range(conv_image_y_axis.shape[1]):
    for j in range(conv_image_y_axis.shape[0]):
        if (conv_image_y_axis[j][i] != 0):
            print("Point detected at:"+str(i)+","+str(j))

            text = str(i) + " , " + str(j)
            cv2.putText(image_colour, text, (i - 40, j + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            cv2.circle(image_colour, (i, j), 10, (0, 255, 0), 1)
            cv2.circle(final_op_image, (i, j), 10, (0, 255, 0), 1)


cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task2/point_detected_labelled.jpg", image_colour)
cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task2/point_detected.jpg", final_op_image)







image_colour = cv2.imread("/Users/vivad/PycharmProjects/CLFinalProject/original_imgs/segment.jpg")
image_gray = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

counter = [0] * 256
for i in range(image_gray.shape[0]):
    for j in range(image_gray.shape[1]):
        counter[image_gray[i, j]]+=1
pixel_range_counter = []
for i in range(0,256):
    pixel_range_counter.append(i)


plt.bar(pixel_range_counter, counter)
plt.ylabel('Intensity')
#plt.show()


def find_thresh(img):
    new_empty_image = np.zeros([img.shape[0], img.shape[1]])
    initial = 210
    #(164, 285) (420, 285) (164, 26) (420, 26)


    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i][j] > initial):
                new_empty_image[i][j] = img[i][j]

    return new_empty_image

def findCorners(img, window_size, k, thresh):

    # X and Y's differential calculator
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = newImg
    offset = window_size // 2

    # Loop through image and find our corners
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Calculate sum of squares
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k * (trace ** 2)

            # If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                # print(x, y, r)
                cornerList.append([x, y, r])

    return color_img, cornerList


orig_segmented_image = find_thresh(image_gray)

corImage, co_ordinates = findCorners(orig_segmented_image, 3, 0.05, 20)

co_ordinates.sort()
left_most = co_ordinates[0]
right_most = co_ordinates[len(co_ordinates) - 1]

top_most = [100, 100, 0]
for array in co_ordinates:
    if (array[1] < top_most[1]):
        top_most = array

bottom_most = [419, 249, 211422285.096875]
for array in co_ordinates:
    if (array[1] > bottom_most[1]):
        bottom_most = array

top_most = tuple(top_most[0:2])
bottom_most = tuple(bottom_most[0:2])
left_most = tuple(left_most[0:2])
right_most = tuple(right_most[0:2])

top_right = []
top_right.append(right_most[0])
top_right.append(top_most[1])
top_right = tuple(top_right)

bottom_right = []
bottom_right.append(right_most[0])
bottom_right.append(bottom_most[1])
bottom_right = tuple(bottom_right)

top_left = []
top_left.append(left_most[0])
top_left.append(top_most[1])
top_left = tuple(top_left)

bottom_left = []
bottom_left.append(left_most[0])
bottom_left.append(bottom_most[1])
bottom_left = tuple(bottom_left)

final_op_image = skimage.color.gray2rgb(orig_segmented_image)
lineThickness = 2
cv2.line(final_op_image, bottom_left, top_left, (0, 255, 255), lineThickness)
cv2.line(final_op_image, bottom_right, top_right, (0, 255, 255), lineThickness)
cv2.line(final_op_image, bottom_left, bottom_right, (0, 255, 255), lineThickness)
cv2.line(final_op_image, top_left, top_right, (0, 255, 255), lineThickness)

cv2.imwrite("/Users/vivad/PycharmProjects/CLFinalProject/Task2/segmented.jpg", final_op_image)

print("The corners of the bounding box are:")
print(bottom_left, bottom_right, top_left, top_right)