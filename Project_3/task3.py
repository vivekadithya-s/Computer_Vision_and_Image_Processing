import numpy as np
import cv2

#   Function to transform the kernel to appropriate type by flipping
def flipping(source):
    source_copy = source.copy()
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            source_copy[i][j] = source[source.shape[0] - i - 1][source.shape[1] - j - 1]
    return source_copy

# Function to perform convolution with sobel operators
def convolution(source_image, kernel_mask):
    kernel_mask = flipping(kernel_mask)
    image_h = source_image.shape[0]
    image_w = source_image.shape[1]
    kernel_h = kernel_mask.shape[0]
    kernel_w = kernel_mask.shape[1]
    h = kernel_h // 2
    w = kernel_w // 2
    convolved_image = np.zeros(source_image.shape)
    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = (sum + kernel_mask[m][n] * source_image[i - h + m][j - w + n])

            convolved_image[i][j] = sum

    return convolved_image

# Function to perform normalization of edge images
def norm(img1, img2):
    img_copy = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q = (img1[i][j] * 2 + img2[i][j] * 2) * (1 / 2)
            if (q > 90):
                img_copy[i][j] = 255
            else:
                img_copy[i][j] = 0
    return img_copy


# Function to detect and draw the hough lines
def draw_hough_lines(img, img2, outfile11, out22, peaks, rhos, thetas):
    #red_col=30
    #blu_col=30
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
        if(pt1[0]!=1):
            if(pt1[0]<0):
                #print(pt1[0])
                #print(a,b)
                cv2.line(img, pt1, pt2, (255,0,0), 3)
                #red_col+=20
            else:
                cv2.line(img2,pt1,pt2,(0,0,255),3)
                #blu_col+=20
    cv2.imwrite(outfile11, img)
    cv2.imwrite(out22, img2)
    return img




# Function to find the acc, rho, and theta params of the hough lines
def find_hough_params(img, rho_res=1, thetas=np.arange(-90, 90, 1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1,1]), 2));
    rhos = np.arange(-rho_max, rho_max, rho_res)
    thetas -= min(min(thetas),0)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img)
    for i in range(len(xis)):
        x = xis[i]
        y = yis[i]
        temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
        temp_rhos = temp_rhos / rho_res + rho_max
        m, n = accumulator.shape
        valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
        temp_rhos = temp_rhos[valid_idxs]
        temp_thetas = thetas[valid_idxs]
        c = np.stack([temp_rhos,temp_thetas], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _,i,counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[i].astype(np.uint)
        accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos


# Function to return max positive values or zero for finding co-ordinates clipping range
def clip(idx):
    return int(max(idx,0))

# Function  to find the peaks for hough transform lines
def detect_peaks_func(H, numpeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((numpeaks,2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        _,max_val,_,max_loc = cv2.minMaxLoc(temp_H)
        if max_val > threshold:
            peaks[i] = max_loc
            (c,r) = max_loc
            t = nhood_size//2.0
            temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:,::-1]



source_im = cv2.imread('/Users/vivad/PycharmProjects/CLFinalProject/original_imgs/hough.jpg', cv2.IMREAD_GRAYSCALE)
copy_source = cv2.imread('/Users/vivad/PycharmProjects/CLFinalProject/original_imgs/hough.jpg', cv2.IMREAD_GRAYSCALE)
x_axis_sobel = np.array([
           [-1,0,1],
           [-2,0,2],
           [-1,0,1]
           ])
y_axis_sobel = np.array([
           [1,2,1],
           [0,0,0],
           [-1,-2,-1]
           ])
img_x_convolved = convolution(source_im, x_axis_sobel)
img_y_convolved = convolution(source_im, y_axis_sobel)
edged_image_source = norm(img_x_convolved, img_y_convolved)
accumulator, theta_values, rho_values = find_hough_params(edged_image_source)
peak_values = detect_peaks_func(accumulator, numpeaks=18, threshold=150, nhood_size=20)
col_copy_1 = cv2.cvtColor(source_im, cv2.COLOR_GRAY2BGR)
col_copy_2 = cv2.cvtColor(copy_source, cv2.COLOR_GRAY2BGR)
draw_hough_lines(col_copy_1, col_copy_2, '/Users/vivad/PycharmProjects/CLFinalProject/Task3/blue_line.jpg', '/Users/vivad/PycharmProjects/CLFinalProject/Task3/red_line.jpg', peak_values, rho_values, theta_values)
