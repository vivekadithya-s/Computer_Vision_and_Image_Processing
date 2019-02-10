UBIT = 'vivekadi'
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(sum([ord(c) for c in UBIT]))
MatchThreshold = 10
import random

#   Reading source images in colour and grayscale respectively
img1col = cv2.imread("data/mountain1.jpg", 0)
img2col = cv2.imread("data/mountain2.jpg", 0)
img1 = cv2.imread("data/mountain1.jpg", 0)
img2 = cv2.imread("data/mountain2.jpg", 0)

#   Keypoint extraction algorithm using Scale Invariant Feature Transform
sift = cv2.xfeatures2d.SIFT_create()

#   Obtaining Keypoints and Descriptor sets from sift's inbuilt detectAndCompute method
(keypoints_set1, description_set1) = sift.detectAndCompute(img1, None)
(keypoints_set2, description_set2) = sift.detectAndCompute(img2, None)

# Drawing the so obtained keypoints on respective colour image
sift_img1 = cv2.drawKeypoints(img1col, keypoints_set1, color=(100, 0, 0), outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(img2col, keypoints_set2, color=(0, 0, 100), outImage=np.array([]))

#print("# kps: {}, descriptors: {}".format(len(keypoints_set1), description_set1.shape))
#print("# kps: {}, descriptors: {}".format(len(keypoints_set2), description_set2.shape))


print("\nSIFT Output generated. File name is: task1_sift1.jpg")
cv2.imwrite("OutputTask1/task1_sift1.jpg", sift_img1)
print("SIFT Output generated. File name is: task1_sift2.jpg")
cv2.imwrite("OutputTask1/task1_sift2.jpg", sift_img2)


#   Keypoint matching using Fast Library for Approximate Nearest Neighbour
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann_module = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann_module.knnMatch(description_set1, description_set2, k=2)

# Storing the best matches as per Lowe's ratio test: m.distance < 0.75 are chosen .
best_matches = []
for m, n in matches:
    #   Checking for best matches as per Lowe's ratio test and appending it accordingly
    if m.distance < 0.75 * n.distance:
        best_matches.append(m)
#   Checking if matches are greater than the minimum number of points required
if (len(best_matches) > MatchThreshold):
    src_pts = np.float32([keypoints_set1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_set2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)


#   Obtaining the Homography matrix and the mask list with True/False values based on inliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #   Computing mask to print all matches, with inliers and outliers
    matchesMask = mask.ravel().tolist()
    #   Computing mask to print only few random inliers
    random_inlierMatchesMask = (mask.ravel()==1).tolist()
    #print(" MAtchesmAsk1 shape")
    #print(random_inlierMatchesMask)
    #print(type(random_inlierMatchesMask))
    #print(len(random_inlierMatchesMask))
    #print(list(matchesMask))
    #print(len(matchesMask))
    #print(matchesMask)

    # Creating a list of random inliers to be printed
    list_temp = []
    for i in range(0, 10):
        rand_val = random.randint(0, 244)
        list_temp.append(rand_val)
    #print(list_t)

    for i in range(0,len(random_inlierMatchesMask)):
        if (i in list_temp):
            ss=0
        else:
            random_inlierMatchesMask[i]=False
    #print(list(random_inlierMatchesMask))
    #print(len(random_inlierMatchesMask))
    h, w = img1.shape
    #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
   # dst = cv2.perspectiveTransform(pts, H)
    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#   Function to warp 2 input images; Returns warped form of both images
def warper_call(img2, img1, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [xmin, ymin] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    image_output = cv2.warpPerspective(img2, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    image_output[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return image_output


warped_image = warper_call(img1col, img2col, H)
cv2.imwrite("OutputTask1/task1_pano.jpg", warped_image)

#   Drawing Parameters for printing image with all matches, both inliers and outliers
drawing_parameters1 = dict(matchColor=(255, 0, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
#   Drawing Parameters for printing image with 10 random matches only for inliers
drawing_parameters2 = dict(matchColor=(0, 0, 255),
                   singlePointColor=None,
                   matchesMask=random_inlierMatchesMask,
                   flags=2)
#   All matches
print("\nImage with all matches - inliers + outliers is created. File name: task1_matches_knn.jpg ")
allmatches = cv2.drawMatches(img1col, keypoints_set1, img2col, keypoints_set2, best_matches, None, **drawing_parameters1)
cv2.imwrite("OutputTask1/task1_matches_knn.jpg", allmatches)

#   Random, n=10, inlier matches
print("\nImage with 10 random matches - only inliers  is created. File name: task1_matches.jpg ")
random_inlier_matches = cv2.drawMatches(img1col, keypoints_set1, img2col, keypoints_set2, best_matches, None, **drawing_parameters2)
cv2.imwrite("OutputTask1/task1_matches.jpg", random_inlier_matches)


print("\nObtained Homography matrix is as follows:")
print(H)

print("task1.py executed")
