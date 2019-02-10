UBIT = 'vivekadi'
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

#   Reading source images in both grayscale and RGB modes
image1col = cv2.imread("data/tsucuba_left.png", 1)
image2col = cv2.imread("data/tsucuba_right.png", 1)
image1 = cv2.imread("data/tsucuba_left.png", 0)
image2 = cv2.imread("data/tsucuba_right.png", 0)

#   SIFT algorithm for obtaining keypoints
sift = cv2.xfeatures2d.SIFT_create()
#   Obtaining Keypoints and Descriptor sets from sift's inbuilt detectAndCompute method

(kps1, descs1) = sift.detectAndCompute(image1, None)
(kps2, descs2) = sift.detectAndCompute(image2, None)

# Drawing the so obtained keypoints on respective colour image
sift_img1 = cv2.drawKeypoints(image1col, kps1, color=(0,255,0), outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(image2col, kps2, color=(0,255,0), outImage=np.array([]))

print("Keypoint count: {}, Descriptors: {}".format(len(kps1), descs1.shape))
print("Keypoint count: {}, Descriptors: {}".format(len(kps2), descs2.shape))

print("\nSIFT Output generated. File name is: task2_sift1.jpg")
cv2.imwrite("OutputTask2/task2_sift1.jpg", sift_img1)
print("\nSIFT Output generated. File name is: task2_sift2.jpg")
cv2.imwrite("OutputTask2/task2_sift2.jpg", sift_img2)

#   Keypoint matching using Fast Library for Approximate Nearest Neighbour
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descs1,descs2,k=2)

# Saving all the best matches as per Lowe's ratio test
good = []
for m,n in matches:
    #   Checking for best matches as per Lowe's ratio test and appending it accordingly

    if m.distance < 0.75*n.distance:
        good.append(m)
#   Random, n=10, inlier matches

src_pts = np.float32([ kps1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good ])
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (0,200,100), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)

print("\nImage with all matches - inliers + outliers is created. File name: task2_matches_knn.jpg ")
img3 = cv2.drawMatches(image1col,kps1,image2col,kps2,good,None,**draw_params)
cv2.imwrite("OutputTask2/task2_matches_knn.jpg", img3)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descs1,descs2,k=2)
print("\nThe Fundamental Matrix obtained is as follows:")
print(F)


#   Disparity image creation
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(112)
stereoMatcher.setBlockSize(17)
stereoMatcher.setSpeckleRange(32)
stereoMatcher.setSpeckleWindowSize(100)
stereo = stereoMatcher.compute(image1, image2)
plt.show(stereo)
plt.imsave("OutputTask2/task2_disparity.png", stereo)
plt.show(stereo)
#print(type(stereo))
#cv2.imshow("image copy", stereo)
#plt.imshow(stereo, interpolation='nearest')

#strmat = np.asmatrix(stereo)
#cv2.imshow("HI",strmat )



src_pts = np.int32(src_pts)
dst_pts = np.int32(dst_pts)

pts1 = src_pts
pts2 = dst_pts


pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        #print("This is pt1" + str(pt2))
        color = tuple(np.random.randint(0,255,3).tolist())
       # print("Color")
        #print(len(color))
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

list = []
for i in range(0,10):
    xx = random.randint(0,271)
    list.append(xx)
#print(list)
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

lines11 = np.copy(lines1)
lines22 = np.copy(lines2)


#print(lines1.shape)
#print(lines11.shape)
count=0
for i in list:
    #print(i)
    lines11[count,:] = lines1[i,:]
    lines22[count,:] = lines2[i,:]

    count = count+1
line112 = np.copy(lines11[1:10])
lines222 = np.copy(lines22[1:10])

#print(line112.shape)
#print(line112)
#print(lines2.shape)

img5,img6 = drawlines(image1,image2,line112,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image



'''
#print("part2")
list_temp = []
for i in range(0,10):
    randnum = random.randint(0,271)
    list_temp.append(randnum)
#print(list_temp)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
#print("lines2 shape")
#print(lines2.shape)
lines22 = np.copy(lines2)
count=0
for i in list_temp:
    #print(i)
    lines22[count,:] = lines2[i,:]
    count = count+1
lines222 = np.copy(lines22[1:10])'''

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
img3,img4 = drawlines(image2,image1,lines222,pts2,pts1)
print(" Epilines images for left and right created as task2_epi_....jpg")
cv2.imwrite("OutputTask2/task2_epi_right.jpg", img5)
cv2.imwrite("OutputTask2/task2_epi_left.jpg", img3)