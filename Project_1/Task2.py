import cv2
import numpy as np
import os
cwd = os.getcwd()

#   Reading the source image in grayscale and RGB mode, please enter Task2 image source
#   path in next 2 lines
#mg_path=cwd+"/task2.jpg"
#print(img_path)
bw = cv2.imread(cwd+"/task2.jpg", 0)
cw = cv2.imread(cwd+"/task2.jpg")

#   Function to find keypoints
def keypoint(imagea, image, imageb, totkp, refac, str1):
    image1=imagea
    image2=imageb
    totcount=totkp
    refactor=refac
    global cw
    global strr
    strr= str1
    img_h = image.shape[0]
    img_w = image.shape[1]
    for i in range(1,img_h-1):
        for j in range(1,img_w-1):
            minimal_flag_count = 0
            maximal_flag_count = 0
            if ( image[i][j] > max( image[i-1][j-1], image[i][j-1], image[i+1][j-1], image[i-1][j], image[i+1][j], image[i-1][j+1], image[i][j+1], image[i+1][j+1]  )     ):
                maximal_flag_count = maximal_flag_count+1
            if ( image[i][j] > max( image1[i-1][j-1], image1[i][j-1], image1[i+1][j-1], image1[i-1][j], image1[i][j], image1[i+1][j], image1[i-1][j+1], image1[i][j+1], image1[i+1][j+1]  )     ):
                maximal_flag_count = maximal_flag_count+1
            if ( image[i][j] > max( image2[i-1][j-1], image2[i][j-1], image2[i+1][j-1], image2[i-1][j], image2[i][j], image2[i+1][j], image2[i-1][j+1], image2[i][j+1], image2[i+1][j+1]  )     ):
                maximal_flag_count = maximal_flag_count+1
            if ( image[i][j] < min( image[i-1][j-1], image[i][j-1], image[i+1][j-1], image[i-1][j], image[i+1][j], image[i-1][j+1], image[i][j+1], image[i+1][j+1]  )     ):
                minimal_flag_count = minimal_flag_count+1
            if ( image[i][j] < min( image1[i-1][j-1], image1[i][j-1], image1[i+1][j-1], image1[i-1][j], image1[i][j], image1[i+1][j], image1[i-1][j+1], image1[i][j+1], image1[i+1][j+1]  )     ):
                minimal_flag_count = minimal_flag_count+1
            if ( image[i][j] < min( image2[i-1][j-1], image2[i][j-1], image2[i+1][j-1], image2[i-1][j], image2[i][j], image2[i+1][j], image2[i-1][j+1], image2[i][j+1], image2[i+1][j+1]  )     ):
                minimal_flag_count = minimal_flag_count+1
            if( minimal_flag_count==0 and maximal_flag_count==3):
                totcount = totcount+1
                print(totcount," keypoint index is (",j*refactor," , ",i*refactor,")   ","\n")
                cw[i*refactor][j*refactor] = 255
                if (((j*refactor)<4) and ((i*refactor)<138)):
                    strr = strr + "\nLeftmost : (" + str(j*refactor) + "," + str(i*refactor) + ")\n"
            elif (minimal_flag_count == 3 and maximal_flag_count == 0):
                totcount = totcount + 1
                print(totcount ," keypoint index is (",j*refactor," , ",i*refactor, ")   ","\n")
                cw[i*refactor][j*refactor] = 255
                if ((j*refactor<4) and ((i*refactor)<138)):
                    strr = strr + "\nLeftmost : (" + str(j*refactor) + "," + str(i*refactor) + ")\n"

#    Function to call the keypoint finder function for different Difference of Gaussian layers in every octave
def formcall():

    #print("Calling1")
    global totkp
    totkp=0
    #str=""
    strrr=""
    str=""
    #   This block of code is used to call the keypoint functions for
    #   Octave 1's DoG 1,2,3 and DoG 2,3,4 respectively
    print("Octave 1 part 1")
    keypoint(dog111, dog222, dog333, totkp, 1, str)
    print(totkp)
    print("Done2")
    strrr=strrr+strr
    print("Octave 1 part 2")
    keypoint(dog222, dog333, dog444, totkp, 1, str)
    print("Done2")

    strrr=strrr+strr

    print("Octave 2 part 1")
    keypoint(dog555, dog666, dog777, totkp, 2, str)
    print(totkp)
    print("DoneO2D123")
    strrr=strrr+strr

    print("Octave 2 part 2")
    keypoint(dog666, dog777, dog888, totkp, 2, str)
    print("DoneO2D234")
    strrr=strrr+strr


    print("Octave 3 part 1")
    keypoint(dog999, dog101010, dog111111, totkp, 4, str)
    print(totkp)
    print("DoneO3D123")
    strrr=strrr+strr

    print("Octave 3 part 2")
    keypoint(dog101010, dog111111, dog121212, totkp, 4, str)
    print("DoneO3D234")
    strrr=strrr+strr


    print("Octave 4 part 1")
    keypoint(dog131313, dog141414, dog151515, totkp, 8, str)
    print(totkp)
    print("DoneO4D123")
    strrr=strrr+strr


    print("Octave 4 part 2")
    keypoint(dog141414, dog151515, dog161616, totkp, 8, str)
    print("DoneO4D234")
    strrr=strrr+strr




    print("Leftmost points are:")
    print(strrr)



    cv2.imshow("All Keypoints", cw)
    cv2.imwrite(cwd+"/FinalImage.png", cw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    h = kheight // 2
    w = kwidth // 2
    #   Initial canvas for the convolved image
    #   using the source image as the basis for size
    convolved_image = np.zeros(image.shape)

    for i in range(h, height - h):
        for j in range(w, width - w):
            sum = 0

            for m in range(kheight):
                for n in range(kwidth):
                    sum = (sum + kernel[m][n] * image[i - h + m][j - w + n])

            convolved_image[i][j] = sum

    return convolved_image







mat_sigma_value = np.zeros(shape=(4, 5))
mat_sigma_value[0, 0] = (1 / (2 ** 0.5))
mat_sigma_value[0, 1] = 1
mat_sigma_value[0, 2] = 2 ** 0.5
mat_sigma_value[0, 3] = 2
mat_sigma_value[0, 4] = 2 * (2 ** 0.5)
mat_sigma_value[1, 0] = 2 ** 0.5
mat_sigma_value[1, 1] = 2
mat_sigma_value[1, 2] = 2 * (2 ** 0.5)
mat_sigma_value[1, 3] = 4
mat_sigma_value[1, 4] = 4 * (2 ** 0.5)
mat_sigma_value[2, 0] = 2 * (2 ** 0.5)
mat_sigma_value[2, 1] = 4
mat_sigma_value[2, 2] = 4 * (2 ** 0.5)
mat_sigma_value[2, 3] = 8
mat_sigma_value[2, 4] = 8 * (2 ** 0.5)
mat_sigma_value[3, 0] = 4 * (2 ** 0.5)
mat_sigma_value[3, 1] = 8
mat_sigma_value[3, 2] = 8 * (2 ** 0.5)
mat_sigma_value[3, 3] = 16
mat_sigma_value[3, 4] = 16 * (2 ** 0.5)


def get_gauss_kernel(l, sig):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))

    return kernel / np.sum(kernel)




#   Function to resize the input image to half the size of
#   original height and width
def image_resize_func(image):
    h, w = image.shape
    factor = 2
    fach = int(h//2)
    facw = int(w//2)

    newim = np.zeros((fach, facw), dtype = int)

    row = 0
    col = 0

    r= 0
    c= 0
    while(row<h-1):
        c = 0
        col = 0
        while(col<w-1):
            newim[r][c] = image[row][col]
            col = col + 2
            c = c + 1
        r = r + 1
        row = row + 2
    return newim


image1 = image_resize_func(bw)
image2 = image_resize_func(image1)
image3 = image_resize_func(image2)

i = 1
rowCount = 1
dcount = 1


#   In this script, we traverse the matrix which has the collection
#   of 20 Sigma values. In each traversal, for each sigma for an octave
#   based on the row count, we obtain the gaussian kernel of size 7x7
#   for the respective sigma value. Next we obtain the gaussian layers
#   by calling the convolution function with the corresponding gaussian layer
#   and grayscale source image to be convolved. Lastly, we find the difference
#   between 2 successive gaussian layers. This is repeatedly for Octave 2, 3, 4 as well
#   in the same fashion as described above.
for row in mat_sigma_value:
    scount = 0
    for sigma in row:

        if (rowCount == 1):

            g = get_gauss_kernel(7, sigma)

            k = convolution(bw, g)
            name = cwd+'/Gauss/Gauss' + str(i) + '.png'

            cv2.imwrite(name, k)
            if (scount >= 1):
                g = k2 - k
                dog = cwd+'/DoG/dog' + str(dcount) + '.png'
                dcount = dcount + 1
                cv2.imwrite(dog, g)
                if( dcount == 2):
                    global dog111, dog222, dog333, dog444
                    dog111 = g
                if (dcount == 3):
                    #global dog222
                    dog222 = g
                if (dcount == 4):
                    #global dog333
                    dog333 = g
                if (dcount == 5):
                    #global dog444
                    dog444 = g
                    #formcall()



            k2 = k
            i = i + 1
            scount = scount + 1

        elif (rowCount == 2):

            g = get_gauss_kernel(7, sigma)

            h = int(bw.shape[0])
            w = int(bw.shape[1])

            small1 = image1

            k = convolution(small1, g)
            name = cwd+'/Gauss/Gauss' + str(i) + '.png'

            cv2.imwrite(name, k)
            if (scount >= 1):
                g = k2 - k
                dog = cwd+'/DoG/dog' + str(dcount) + '.png'
                dcount = dcount + 1
                cv2.imwrite(dog, g)
                cv2.imwrite(dog, g)
                if (dcount == 6):
                    global dog555, dog666, dog777, dog888
                    dog555 = g
                if (dcount == 7):
                    dog666 = g
                if (dcount == 8):
                    # global dog333
                    dog777 = g
                if (dcount == 9):
                    # global dog444
                    dog888 = g
                    #formcall()

            k2 = k
            cv2.imwrite(name, k)
            i = i + 1
            scount = scount + 1

        elif (rowCount == 3):

            g = get_gauss_kernel(7, sigma)

            h = int(bw.shape[0])
            w = int(bw.shape[1])

            small2 = image2

            k = convolution(small2, g)
            name = cwd+'/Gauss/Gauss' + str(i) + '.png'

            cv2.imwrite(name, k)
            if (scount >= 1):
                g = k2 - k
                dog = cwd+'/DoG/dog' + str(dcount) + '.png'
                dcount = dcount + 1
                cv2.imwrite(dog, g)
                if (dcount == 10):
                    global dog999, dog101010, dog111111, dog121212
                    dog999 = g
                if (dcount == 11):
                    # global dog222
                    dog101010 = g
                if (dcount == 12):
                    # global dog333
                    dog111111 = g
                if (dcount == 13):
                    # global dog444
                    dog121212 = g
                    #formcall()

            k2 = k
            cv2.imwrite(name, k)
            i = i + 1
            scount = scount + 1

        else:

            g = get_gauss_kernel(7, sigma)

            h = int(bw.shape[0])
            w = int(bw.shape[1])

            small3 = image3
            k = convolution(small3, g)
            name = cwd+'/Gauss/Gauss' + str(i) + '.png'
            cv2.imwrite(name, k)
            if (scount >= 1):
                g = k2 - k
                dog = cwd+'/DoG/dog' + str(dcount) + '.png'
                dcount = dcount + 1

                cv2.imwrite(dog, g)
                if (dcount == 14):
                    global dog131313, dog141414, dog151515, dog161616
                    dog131313 = g
                if (dcount == 15):
                    # global dog222
                    dog141414 = g
                if (dcount == 16):
                    # global dog333
                    dog151515 = g
                if (dcount == 17):
                    # global dog444
                    dog161616 = g
                    formcall()

            k2 = k
            cv2.imwrite(name, k)
            i = i + 1
            scount = scount + 1

    rowCount = rowCount + 1
