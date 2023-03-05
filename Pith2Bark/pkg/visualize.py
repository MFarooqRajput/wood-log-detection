import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkg.model import is_grayscale
from pkg.model import is_polar
from pkg.model import is_polar_edge
from pkg.model import is_polar_pith_edge

def plot_image(image):
    plt.imshow(image)
    plt.show()

def plot_gray_image(gray_image):
    plt.imshow(gray_image, cmap='gray')
    plt.show()

def plot_radial_lines(model, gray_image, polar_image, edge_image, prediction, width, height, points_on_contour):
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

def plot_rings_count(algos, sources, orig_mean, no_of_lines):
    for i in range(len(algos)):
        source = sources[i]
        source_mean = np.mean(source)
        name = algos[i]
        title = "Line Vs Rings Per Line (" + name + ")"
        plt.bar(np.arange(start=1, stop=no_of_lines+1, step=1), source)
        plt.title(title)
        plt.xlabel('Lines')
        plt.ylabel('Rings')
        plt.xticks(np.arange(0, no_of_lines+1, 1))
        plt.axhline(y=orig_mean, color='g', linestyle='-')
        plt.axhline(y=source_mean, color='r', linestyle='-')
        plt.show()

def plot_min_max_mean_orig(algos, sources, no_of_images):
    for i in range(len(algos)):
        source = sources[i]
        name = algos[i]
        title = "Images Vs Rings (" + name + ")"
        
        source = np.array(source)
        x = np.arange(start=1, stop=len(source) + 1, step=1)
        min_rings = source[:, 0]
        max_rings = source[:, 1]
        mean_rings = source[:, 2]
        orignal_rings = source[:, 3]
        width = 0.2

        x_labels = []
        for idx in range(no_of_images):
            x_labels.append("Img" + str(idx))

        # plot data in grouped manner of bar type
        plt.title(title)
        plt.bar(x-0.3, min_rings, width, color='red')
        plt.bar(x-0.1, max_rings, width, color='blue')
        plt.bar(x+0.1, mean_rings, width, color='limegreen')
        plt.bar(x+0.3, orignal_rings, width, color='green')
        plt.xticks(x, x_labels)
        plt.xlabel("Images")
        plt.ylabel("Rings")
        plt.legend(["Min", "Max", "Mean", "Orignal"])
        plt.show()

def plot_mse(x, y):
    plt.bar(x, y, width = 0.1)
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.axhline(y=min(y), color='g', linestyle='-')
    plt.show()

def plot_multi_mse(mse_url, show_group=True):
    
    df = pd.read_csv(mse_url, index_col='model')
    col_list = list(df)
    min_mse = df[col_list].min(axis=1).min()
    max_mse = df[col_list].max(axis=1).max()
    winner_idx = df[col_list].min(axis=1).idxmin()
    
    no_of_bins = max_mse // 50

    print("%s %s (%s) \n" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin()))

    print(df)
    print("Min: %s, Max: %s" % (min_mse, max_mse))

    df.plot(kind="bar")

    plt.xticks(rotation=45, horizontalalignment="right")
    plt.title("Model vs MSE")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.axhline(y=min_mse, color='g', linestyle='--', linewidth=1)
    plt.locator_params(axis='y', nbins=no_of_bins)
    plt.show()

    if show_group:
        # three groups
        df1 = df.iloc[:15,:]
        df2 = df.iloc[15:27,:]
        df3 = df.iloc[27:,:]
        df_arr = []
        df_arr.append(df1)
        df_arr.append(df2)
        df_arr.append(df3)

        for i in range(len(df_arr)):
            df = df_arr[i]
            col_list = list(df)
            min_mse = df[col_list].min(axis=1).min()
            max_mse = df[col_list].max(axis=1).max()
            no_of_bins = max_mse // 50
            print(df)
            print("Min: %s, Max: %s" % (min_mse, max_mse))

            df.plot(kind="bar")

            plt.xticks(rotation=45, horizontalalignment="right")
            plt.title("Model vs MSE")
            plt.xlabel("Model")
            plt.ylabel("MSE")
            plt.axhline(y=min_mse, color='g', linestyle='--', linewidth=1)
            plt.locator_params(axis='y', nbins=no_of_bins)
            plt.show()