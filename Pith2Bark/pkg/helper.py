import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def append_dict_to_df(df,dict_to_append):

    return pd.concat([df, pd.DataFrame.from_records([dict_to_append])], ignore_index=True)

def merge_df(left_df, right_df, left_col, right_col):

    return pd.merge(left=left_df, right=right_df, left_on=left_col, right_on=right_col)

def is_image_file(filename):

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    # store the images name and index here!
    images_df = pd.DataFrame(columns = ['image_index', 'image_name'])

    # store the images here!
    images = [] 
    nr_of_images = 0
    for fname in os.listdir(path):
        if is_image_file(fname):
            image = plt.imread(os.path.join(path, fname))
            images.append(image)
            images_df = append_dict_to_df(images_df,{'image_index' : nr_of_images, 'image_name' : fname})
            nr_of_images += 1
    assert images, '{:s} has no valid image file'.format(path)

    return images, images_df

def get_contours(gray_image):
    ret, thresh = cv2.threshold(gray_image, 1, 255, 1) #cv2.threshold(gray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = cv2.drawContours(gray_image, contours, -1, 255, 3) #cv2.drawContours(img, contours, -1, (255,0,0), 3)
        
    contours_array = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours_array.append(contours[i][j][0])
    
    return contours_image, contours_array

def get_points_on_contour(contours_array, no_of_lines):
    #point_indexes = np.random.randint(0, len(contours_array)-1, size = no_of_lines)
    
    contours_array.sort(key=lambda c:math.atan2(c[0], c[1]))
    n = len(contours_array) - 1
    k = no_of_lines
    point_indexes = [i * (n // k) + min(i, n % k) for i in range(k)]
    
    points_on_contour = []
    for index in point_indexes:
        points_on_contour.append(contours_array[index])

    #points_on_contour.sort(key=lambda c:math.atan2(c[0], c[1]))

    return points_on_contour

def pixel_to_image(pixel_values):
    tile_pixel_values = np.tile(pixel_values, (50, 1))
    frame = cv2.adaptiveThreshold(tile_pixel_values, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return frame

def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

def convert_to_polar(gray_image):
    value = np.sqrt(((gray_image.shape[0]/2.0)**2.0)+((gray_image.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(gray_image,(gray_image.shape[0]/2, gray_image.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)

    return polar_image 

def convert_to_polar_pith(gray_image, prediction):
    ro,col = gray_image.shape
    cent = [prediction[0]*ro, prediction[1]*col]
    value = np.sqrt(((gray_image.shape[0]/2.0)**2.0)+((gray_image.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(gray_image, cent, value, cv2.WARP_FILL_OUTLIERS)

    return polar_image

def convert_to_sobel_edge(gray_image, k=3):
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobel_edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return sobel_edge_image

def convert_to_sobel_edge_blur(gray_image, k=3):
    gray_image = cv2.GaussianBlur(gray_image, (3,3), 0) 
    
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobel_edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return sobel_edge_image 

def convert_to_canny_edge(gray_image, t1=60, t2=180):
    canny_edge_image = cv2.Canny(gray_image, t1, t2)

    return canny_edge_image
    
def convert_to_canny_edge_otsu(gray_image):
    otsu_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    otsu_thresh = get_range(otsu_thresh)
    edge_otsu = cv2.Canny(gray_image, *otsu_thresh)

    return edge_otsu

def convert_to_canny_edge_triangle(gray_image):
    triangle_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRIANGLE)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    triangle_thresh = get_range(triangle_thresh)
    edge_triangle = cv2.Canny(gray_image, *triangle_thresh)
    
    return edge_triangle

def convert_to_canny_edge_manual(gray_image):
    manual_thresh = np.median(gray_image)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    manual_thresh = get_range(manual_thresh)
    edge_manual = cv2.Canny(gray_image, *manual_thresh)
    
    return edge_manual

def convert_to_canny_edge_blur(gray_image,t1=60,t2=180):
    img_blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    canny_edge_image = cv2.Canny(img_blur, t1, t2)
    
    return canny_edge_image

def convert_to_canny_edge_blur_otsu(gray_image):
    img_blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    otsu_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    otsu_thresh = get_range(otsu_thresh)
    edge_otsu = cv2.Canny(img_blur, *otsu_thresh)
    
    return edge_otsu
    
def convert_to_canny_edge_blur_triangle(gray_image):
    img_blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    triangle_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_TRIANGLE)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    triangle_thresh = get_range(triangle_thresh)
    edge_triangle = cv2.Canny(img_blur, *triangle_thresh)
    
    return edge_triangle

def convert_to_canny_edge_blur_manual(gray_image):
    img_blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    manual_thresh = np.median(img_blur)

    def get_range(threshold, sigma=0.33):
            return (1-sigma) * threshold, (1+sigma) * threshold

    manual_thresh = get_range(manual_thresh)
    edge_manual = cv2.Canny(img_blur, *manual_thresh)
    
    return edge_manual

def convert_to_laplacian_edge(gray_image, k=3):
    laplacian = cv2.Laplacian(gray_image, 5, cv2.CV_64F, ksize=k)
    laplacian_edge_image = cv2.convertScaleAbs(laplacian)

    return laplacian_edge_image

def convert_to_laplacian_edge_blur(gray_image, k=3):
    gray_image = cv2.GaussianBlur(gray_image, (3,3), 0) 
    
    laplacian = cv2.Laplacian(gray_image, 5, cv2.CV_64F, ksize=k)
    laplacian_edge_image = cv2.convertScaleAbs(laplacian)
    
    return laplacian_edge_image 