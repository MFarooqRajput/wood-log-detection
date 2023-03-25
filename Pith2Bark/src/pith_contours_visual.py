import matplotlib.pyplot as plt

from src.helper import convert_to_polar
from src.helper import convert_to_polar_pith

def plot_image_with_pith(image, prediction):
    height, width = image.shape[0], image.shape[1]
    
    plt.imshow(image)
    plt.plot(prediction[0]*width, prediction[1]*height, '.')
    plt.show()

def plot_gray_polar_pith_images(grayscale_images, predictions):
    rows = len(grayscale_images)
    columns = 3

    fig = plt.figure(figsize=(columns*4,rows*4))
    col_index = 1

    for idx, gray_image in enumerate(grayscale_images):
        prediction = predictions[idx]
        polar_image = convert_to_polar(gray_image)
        polar_pith_image = convert_to_polar_pith(gray_image, prediction)
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        
        fig.add_subplot(rows, columns, col_index+1)
        plt.imshow(polar_image, cmap='gray')
        plt.axis('off')
        
        fig.add_subplot(rows, columns, col_index+2)
        plt.imshow(polar_pith_image, cmap='gray')
        plt.axis('off')
        
        col_index += 3

    plt.show()

def plot_contours_with_points(contours_image, points_on_contour):
    plt.imshow(contours_image)

    for point_on_contour in points_on_contour:
        plt.plot(point_on_contour[0], point_on_contour[1], '.', color = 'r', markersize = 10) 
            
    plt.show()