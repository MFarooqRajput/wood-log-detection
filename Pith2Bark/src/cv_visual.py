import matplotlib.pyplot as plt
import numpy as np

from src.cv_model import is_grayscale
from src.cv_model import is_polar
from src.cv_model import is_polar_edge
from src.cv_model import is_polar_pith_edge

def plot_radial_lines(model, gray_image, polar_image, edge_image, prediction, points_on_contour):
    height, width = gray_image.shape[0], gray_image.shape[1]

    if is_grayscale(model):
        plt.imshow(gray_image, cmap = 'gray')
    elif is_polar(model):
        plt.imshow(polar_image, cmap = 'gray')
    elif is_polar_edge(model) or is_polar_pith_edge(model):
        plt.imshow(edge_image, cmap = 'gray')
    else:
        plt.imshow(edge_image, cmap = 'gray')

    if is_polar(model) or is_polar_edge(model) or is_polar_pith_edge(model):
        pass
    else:
        plt.plot(prediction[0]*width, prediction[1]*height, '.', markersize = 10)
    
    for point_on_contour in points_on_contour:
        plt.plot(point_on_contour[0], point_on_contour[1], '.', color = 'r', markersize = 10)

        if is_polar(model) or is_polar_edge(model) or is_polar_pith_edge(model):
            plt.plot(np.array([point_on_contour[0], 0]), np.array([point_on_contour[1], point_on_contour[1]]))
        else:
            plt.plot(np.array([point_on_contour[0], prediction[0]*width]), np.array([point_on_contour[1], prediction[1]*height]))
            
    plt.show()

def plot_lines(model, gray_image, polar_image, edge_image, lines):

    if is_grayscale(model):
        plt.imshow(gray_image, cmap = 'gray')
    elif is_polar(model):
        plt.imshow(polar_image, cmap = 'gray')
    elif is_polar_edge(model) or is_polar_pith_edge(model):
        plt.imshow(edge_image, cmap = 'gray')
    else:
        plt.imshow(edge_image, cmap = 'gray')

    for line in lines:
        plt.plot(line[:,0], line[:,1], '.')
    plt.show()

def plot_change_along_radial_lines(lines_pixel_values):
    for pixel_values in lines_pixel_values:
        plt.plot(np.linspace(0,len(pixel_values),len(pixel_values)), pixel_values)
    plt.xlabel('point from estimated pith location to contour')
    plt.ylabel('pixel value')
    plt.show()

def plot_rings_along_lines(title, x, y, orig_mean):
    plt.bar(x, y)
    
    plt.title("Line Vs Rings Per Line (" + title + ")")
    plt.xlabel('Lines')
    plt.ylabel('Rings')
    plt.axhline(y=orig_mean, color='g', linestyle='--', linewidth=1)
    plt.axhline(y=np.mean(y), color='r', linestyle='--', linewidth=1)
    plt.xticks(np.arange(0, len(x)+1, 1))
    plt.show()

def plot_min_max_mean_orig(title, x, y):
    plt.bar(x-0.3, y[:, 0], width = 0.2, color='red')
    plt.bar(x-0.1, y[:, 1], width = 0.2, color='blue')
    plt.bar(x+0.1, y[:, 2], width = 0.2, color='limegreen')
    plt.bar(x+0.3, y[:, 3], width = 0.2, color='green')

    plt.title("Images Vs Rings (" + title + ")")
    plt.xlabel("Images")
    plt.ylabel("Rings")
    plt.legend(["Min", "Max", "Mean", "Orignal"])
    plt.xticks(np.arange(0, len(x)+1, 1))
    plt.show()

def plot_rmse(x, y):
    plt.bar(x, y, width = 0.2)
    
    plt.title("Model vs RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.axhline(y=min(y), color='g', linestyle='--', linewidth=1)
    plt.show()