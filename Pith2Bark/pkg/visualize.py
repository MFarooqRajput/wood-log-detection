import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as matplot
import seaborn as sns

from pkg.model import is_grayscale, is_polar, is_polar_edge, is_polar_pith_edge
from pkg.helper import convert_to_polar, convert_to_polar_pith

def plot_image(image):
    plt.imshow(image)
    plt.show()

def plot_gray_image(gray_image):
    plt.imshow(gray_image, cmap='gray')
    plt.show()

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

def plot_mse(x, y):
    plt.bar(x, y, width = 0.2)
    
    plt.title("Model vs MSE")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.axhline(y=min(y), color='g', linestyle='--', linewidth=1)
    plt.show()

def plot_multi_mse(df, min_mse, no_of_bins):
    df.plot(kind="bar")
    
    plt.title("Model vs MSE")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.axhline(y=min_mse, color='g', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.locator_params(axis='y', nbins=no_of_bins)
    plt.show()

def plot_images(titles, images):
    rows = 13
    columns = 3

    fig = plt.figure(figsize=(columns*4,rows*4))
    col_index = 1

    for i in range(3):
        image = images[i]
        title = titles[i]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        col_index += 1

    for i in range(12):
        image = images[i+3]
        title = titles[i+3]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        image = images[i+15]
        title = titles[i+15]
        
        fig.add_subplot(rows, columns, col_index+1)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        image = images[i+27]
        title = titles[i+27]
        
        fig.add_subplot(rows, columns, col_index+2)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        col_index += 3

    plt.show()

def plot_images_group(titles, images, rows, columns):
    fig = plt.figure(figsize=(columns*4,rows*4))
    col_index = 1

    for i in range(len(titles)):
        image = images[i]
        title = titles[i]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')

        col_index += 1

    plt.show()

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

def heat_map(df, max=1, min=-1):

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df, dtype=bool),1)

    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))
    f, ax = plt.subplots(figsize=(4, 3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    labels = ['Not Damage', 'Damaged']
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, mask=mask, cmap=cmap, vmax=max, center=(max+min)/2, vmin=min,
                square=True, annot=True, linewidths=.2, cbar_kws={"shrink": .5},
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.show()

def plot_accuracy(x, y, tick_label):
    plt.bar(x, y, tick_label=tick_label, width=0.8)
    
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model vs Accuracy')
    plt.show()