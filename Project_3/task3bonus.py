import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

# Function to perform Gaussian Smoothing
def smoothing_func(input_img):
    gauss_filter = np.array([[0.109, 0.111, 0.109], [0.111, 0.135, 0.111], [0.109, 0.111, 0.109]])
    return cv2.filter2D(input_img, -1, gauss_filter)

# Function to perform convolution transformation or flipping of kernel
def flipping_func(source_im):
    source_copy = source_im.copy()
    for i in range(source_im.shape[0]):
        for j in range(source_im.shape[1]):
            source_copy[i][j] = source_im[source_im.shape[0] - i - 1][source_im.shape[1] - j - 1]
    return source_copy

#   Function to perform the actual convolution to find edges
def conv(source_im, ker_matrix):
    ker_matrix = flipping_func(ker_matrix)
    image_h = source_im.shape[0]
    image_w = source_im.shape[1]

    kernel_h = ker_matrix.shape[0]
    kernel_w = ker_matrix.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(source_im.shape)

    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = (sum + ker_matrix[m][n] * source_im[i - h + m][j - w + n])

            image_conv[i][j] = sum

    return image_conv

# Function to perform normalization of edge images
def normalization_func(src_a, src_b):
    src_a_copy = np.zeros(src_a.shape)

    for i in range(src_a.shape[0]):
        for j in range(src_a.shape[1]):
            q = (src_a[i][j] * 2 + src_b[i][j] * 2) * (1 / 2)
            if (q > 90):
                src_a_copy[i][j] = 255
            else:
                src_a_copy[i][j] = 0

    return src_a_copy

# Function to detect circles using hough transform and to return x,y co-ordinates with radius values
def detect_circle_params(source, list_of_circs):
    rows = source.shape[0]
    cols = source.shape[1]


    sinang = dict()
    cosang = dict()


    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)



    radius = [i for i in range(19, 24)]

    for r in radius:
        accumulator_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        for x in range(rows):
            for y in range(cols):
                if source[x][y] == 255:
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            a1 = int(a)
                            b1 = int(b)

                            accumulator_cells[a1][b1] += 1

        accumulator__max = np.amax(accumulator_cells)

        if (accumulator__max > 150):


            accumulator_cells[accumulator_cells < 150] = 0

            for row_p in range(rows):
                for col_p in range(cols):
                    if (row_p > 0 and col_p > 0 and row_p < rows - 1 and col_p < cols - 1 and accumulator_cells[row_p][col_p] >= 150):
                        sum_of_averages = np.float32((accumulator_cells[row_p][col_p] + accumulator_cells[row_p - 1][col_p] + accumulator_cells[row_p + 1][col_p] +
                                              accumulator_cells[row_p][col_p - 1] + accumulator_cells[row_p][col_p + 1] + accumulator_cells[row_p - 1][col_p - 1] +
                                              accumulator_cells[row_p - 1][col_p + 1] + accumulator_cells[row_p + 1][col_p - 1] + accumulator_cells[row_p + 1][
                                                  col_p + 1]) / 9)
                        if (sum_of_averages >= 50):
                            list_of_circs.append((row_p, col_p, r))
                            accumulator_cells[row_p:row_p + 5, col_p:col_p + 7] = 0


source_path = '/Users/vivad/Downloads/original_imgs/hough.jpg'

source_images = cv2.imread(source_path)
sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

input_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

input_img_copy = input_image.copy()

#Performing convolution in x axis
x_convolved = conv(input_img_copy, sobel_x)

#Performing convolution in y axis
y_convolved = conv(input_img_copy, sobel_y)

#Performing convolution in for both x axis and y axis
edges_only = normalization_func(x_convolved, y_convolved)

# List to store x,y and radius values of detected circles
list_of_circs = []

#Function to compute the above list of circles' params
detect_circle_params(edges_only, list_of_circs)

for point in list_of_circs:
    cv2.circle(source_images, (point[1], point[0]), point[2], (50, 0, 255), 1)
cv2.imwrite('/Users/vivad/Downloads/CVIPOUT/Task3/coins_output.jpg', source_images)


