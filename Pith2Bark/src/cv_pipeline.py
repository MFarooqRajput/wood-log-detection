import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error

from src.cv_algorithms import subseq
from src.cv_algorithms import peaks
from src.cv_algorithms import binary
from src.cv_algorithms import moving_window_averages

from src.cv_helper import read_images
from src.cv_helper import merge_df
from src.cv_helper import convert_to_grayscale
from src.cv_helper import convert_to_polar
from src.cv_helper import convert_to_polar_pith
from src.cv_helper import convert_to_sobel_edge
from src.cv_helper import convert_to_sobel_edge_blur
from src.cv_helper import convert_to_canny_edge
from src.cv_helper import convert_to_canny_edge_otsu
from src.cv_helper import convert_to_canny_edge_triangle
from src.cv_helper import convert_to_canny_edge_manual
from src.cv_helper import convert_to_canny_edge_blur
from src.cv_helper import convert_to_canny_edge_blur_otsu
from src.cv_helper import convert_to_canny_edge_blur_triangle
from src.cv_helper import convert_to_canny_edge_blur_manual
from src.cv_helper import convert_to_laplacian_edge
from src.cv_helper import convert_to_laplacian_edge_blur
from src.cv_helper import get_contours
from src.cv_helper import get_points_on_contour
from src.cv_helper import pixel_to_image

from src.cv_model import Model
from src.cv_model import is_grayscale
from src.cv_model import is_polar
from src.cv_model import is_polar_edge
from src.cv_model import is_polar_pith_edge

from src.cv_pith_prediction import get_pith_prediction

from src.cv_read_rings import read_rings_ranking
from src.cv_read_rings import read_rings_count

from src.cv_visual import plot_radial_lines
from src.cv_visual import plot_lines
from src.cv_visual import plot_change_along_radial_lines
from src.cv_visual import plot_rings_along_lines
from src.cv_visual import plot_min_max_mean_orig
from src.cv_visual import plot_rmse

ALGOS = ["subseq", "peaks", "binary", "mwa"]

def prepare_dataset(data_dir, ranking_fname, pickle_fname):

    try:
        images_ranking_prediction_df = pd.read_pickle(pickle_fname)

        # get images
        images, images_df = read_images(data_dir)

    except Exception as e:
        
        # print exception
        #print(e)

        # get images
        images, images_df = read_images(data_dir)

        # get ranking
        if "ringsranking" in ranking_fname:
            ranking_df = read_rings_ranking(ranking_fname)
        else:
            ranking_df = read_rings_count(ranking_fname)

        # ranking with images
        images_ranking_df = merge_df(ranking_df, images_df, 'image_name', 'image_name')

        # pith prediction
        prediction_df = get_pith_prediction(images)

        # ranking of images with pith prediction
        images_ranking_prediction_df = merge_df(images_ranking_df, prediction_df, 'image_index', 'image_index')

        # save
        images_ranking_prediction_df.to_pickle(pickle_fname)

    return images, images_ranking_prediction_df

def get_prediction_ranking(idx, images_ranking_prediction_df):
    df = images_ranking_prediction_df.loc[images_ranking_prediction_df['image_index'] == idx]
    image_name = str(df['image_name'].values[0])
    prediction = df['prediction'].to_numpy()[0]
    ranking = float(df['ranking'].values[0])

    return image_name, prediction, ranking

def process_image(model, prediction, image):

    gray_image = convert_to_grayscale(image)

    if model == Model.GRAYSCALE:
        return gray_image, None, None
    elif model == Model.POLAR:
        polar_image = convert_to_polar(gray_image)
        return gray_image, polar_image, None
    elif model == Model.POLAR_PITH:
        polar_image = convert_to_polar_pith(gray_image, prediction)
        return gray_image, polar_image, None
    else:
        pass

    if model == Model.SOBEL_EDGE:
        sobel_edge_image = convert_to_sobel_edge(gray_image)
        return gray_image, None, sobel_edge_image
    elif model == Model.SOBEL_EDGE_BLUR:
        sobel_edge_image = convert_to_sobel_edge_blur(gray_image)
        return gray_image, None, sobel_edge_image
    elif model == Model.CANNY_EDGE:
        canny_edge_image = convert_to_canny_edge(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_OTSU:
        canny_edge_image = convert_to_canny_edge_otsu(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_triangle(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_MANUAL:
        canny_edge_image = convert_to_canny_edge_manual(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_BLUR:
        canny_edge_image = convert_to_canny_edge_blur(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_BLUR_OTSU:
        canny_edge_image = convert_to_canny_edge_blur_otsu(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_BLUR_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_blur_triangle(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.CANNY_EDGE_BLUR_MANUAL:
        canny_edge_image = convert_to_canny_edge_blur_manual(gray_image)
        return gray_image, None, canny_edge_image
    elif model == Model.LAPLACIAN_EDGE:
        laplacian_edge_image = convert_to_laplacian_edge(gray_image)
        return gray_image, None, laplacian_edge_image
    elif model == Model.LAPLACIAN_EDGE_BLUR:
        laplacian_edge_image = convert_to_laplacian_edge_blur(gray_image)
        return gray_image, None, laplacian_edge_image
    else:
        pass

    polar_image = convert_to_polar(gray_image)

    if model == Model.POLAR_SOBEL_EDGE:
        sobel_edge_image = convert_to_sobel_edge(polar_image)
        return gray_image, polar_image, sobel_edge_image
    elif model == Model.POLAR_SOBEL_EDGE_BLUR:
        sobel_edge_image = convert_to_sobel_edge_blur(polar_image)
        return gray_image, polar_image, sobel_edge_image
    elif model == Model.POLAR_CANNY_EDGE:
        canny_edge_image = convert_to_canny_edge(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_OTSU:
        canny_edge_image = convert_to_canny_edge_otsu(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_triangle(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_MANUAL:
        canny_edge_image = convert_to_canny_edge_manual(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_BLUR:
        canny_edge_image = convert_to_canny_edge_blur(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_BLUR_OTSU:
        canny_edge_image = convert_to_canny_edge_blur_otsu(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_BLUR_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_blur_triangle(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_CANNY_EDGE_BLUR_MANUAL:
        canny_edge_image = convert_to_canny_edge_blur_manual(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_LAPLACIAN_EDGE:
        laplacian_edge_image = convert_to_laplacian_edge(polar_image)
        return gray_image, polar_image, laplacian_edge_image
    elif model == Model.POLAR_LAPLACIAN_EDGE_BLUR:
        laplacian_edge_image = convert_to_laplacian_edge_blur(polar_image)
        return gray_image, polar_image, laplacian_edge_image
    else:
        pass

    polar_image = convert_to_polar_pith(gray_image, prediction)

    if model == Model.POLAR_PITH_SOBEL_EDGE:
        sobel_edge_image = convert_to_sobel_edge(polar_image)
        return gray_image, polar_image, sobel_edge_image
    elif model == Model.POLAR_PITH_SOBEL_EDGE_BLUR:
        sobel_edge_image = convert_to_sobel_edge_blur(polar_image)
        return gray_image, polar_image, sobel_edge_image
    elif model == Model.POLAR_PITH_CANNY_EDGE:
        canny_edge_image = convert_to_canny_edge(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_OTSU:
        canny_edge_image = convert_to_canny_edge_otsu(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_triangle(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_MANUAL:
        canny_edge_image = convert_to_canny_edge_manual(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_BLUR:
        canny_edge_image = convert_to_canny_edge_blur(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_BLUR_OTSU:
        canny_edge_image = convert_to_canny_edge_blur_otsu(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_BLUR_TRIANGLE:
        canny_edge_image = convert_to_canny_edge_blur_triangle(polar_image)
        return gray_image, polar_image, canny_edge_image 
    elif model == Model.POLAR_PITH_CANNY_EDGE_BLUR_MANUAL:
        canny_edge_image = convert_to_canny_edge_blur_manual(polar_image)
        return gray_image, polar_image, canny_edge_image
    elif model == Model.POLAR_PITH_LAPLACIAN_EDGE:
        laplacian_edge_image = convert_to_laplacian_edge(polar_image)
        return gray_image, polar_image, laplacian_edge_image
    elif model == Model.POLAR_PITH_LAPLACIAN_EDGE_BLUR:
        laplacian_edge_image = convert_to_laplacian_edge_blur(polar_image)
        return gray_image, polar_image, laplacian_edge_image    
    else:
        pass

def change_along_radial_lines(model, gray_image, polar_image, edge_image, prediction, no_of_lines, no_of_points):

    source_image = gray_image

    if is_polar(model) or is_polar_edge(model) or is_polar_pith_edge(model):
        source_image = polar_image

    height, width = source_image.shape[0], source_image.shape[1]
    
    # contours
    contours_image, contours_array = get_contours(source_image)

    # contours points
    points_on_contour = get_points_on_contour(contours_array, no_of_lines)

    lines_len = []
    lines = []
    lines_pixel_values = []

    for point_on_contour in points_on_contour:

        if is_polar(model) or is_polar_edge(model) or is_polar_pith_edge(model):
            line_len = math.dist(np.array([0, point_on_contour[1]]), point_on_contour)
            no_of_points = int(line_len)
            line = np.linspace(np.array([0, point_on_contour[1]]), point_on_contour, no_of_points)
        
        else:
            line_len = math.dist(np.array([prediction[0]*width, prediction[1]*height]), point_on_contour)
            no_of_points = int(line_len)
            line = np.linspace(np.array([prediction[0]*width, prediction[1]*height]), point_on_contour, no_of_points)

        # we convert to integer numbers
        for i in range(len(line)):
            for j in range(2):
                line[i,j] = int(line[i,j])

        line = line.astype('int') # we do this for proper indexing

        if is_grayscale(model):
            pass
        elif is_polar(model):
            pass
        elif is_polar_edge(model) or is_polar_pith_edge(model):
            source_image = edge_image
        else:
            source_image = edge_image

        pixel_values = []
        i = 0
        while i < len(line):
            value = source_image[line[i][1], line[i][0]] 
            pixel_values.append(value)
            i += 1
        
        lines_len.append(line_len)
        lines.append(line)
        lines_pixel_values.append(pixel_values)
    '''
    #TRY
    print("TRY")
    if is_polar(model) or is_polar_edge(model) or is_polar_pith_edge(model):
        #get index of points_on_contour
        for point_on_contour in points_on_contour:
            #index = contours_array.index(point_on_contour).all()
            target_indices = [index for index,point in enumerate(contours_array) if point[0]==point_on_contour[0] and point[1]==point_on_contour[1]]
            index  = target_indices[0]
            #get +/-4
            point_on_contour_arr = contours_array[index - 4: index + 4]

            pixel_values_arr = [[], [], [], [], [], [], [], []]
            pixel_values_arr_i = 0
            for point_on_contour_s in point_on_contour_arr:

                line_len_s = math.dist(np.array([0, point_on_contour_s[1]]), point_on_contour_s)
                req_points_s = int(line_len_s) #change to max points along line
                line_s = np.linspace(np.array([0, point_on_contour_s[1]]), point_on_contour_s, req_points_s)

                #we convert to integer numbers
                for i in range(len(line_s)):
                    for j in range(2):
                        line_s[i,j] = int(line_s[i,j])

                line_s = line_s.astype('int') #we do this for proper indexing

                pixel_values_s = []
                i = 0
                while i < len(line_s):
                    value = source_image[line_s[i][1], line_s[i][0]] 
                    pixel_values_s.append(value)
                    i += 1

                pixel_values_arr[pixel_values_arr_i] = pixel_values_s
                pixel_values_arr_i += 1
            
            len_each_line = [len(part) for part in pixel_values_arr]
            min_len = min(len_each_line)
            pixel_values = []

            for idx in range(min_len):
                p1 = pixel_values_arr[0][idx]
                p2 = pixel_values_arr[1][idx]
                p3 = pixel_values_arr[2][idx]
                p4 = pixel_values_arr[3][idx]
                p5 = pixel_values_arr[4][idx]
                p6 = pixel_values_arr[5][idx]
                p7 = pixel_values_arr[6][idx]
                p8 = pixel_values_arr[7][idx]

                p_p = int(p1) + int(p2) + int(p3) + int(p4) + int(p5) + int(p6) + int(p7) + int(p8)
                p = p_p / 8
                p_v = 255 if p > 140 else 0
                
                pixel_values.append(p_v)
            lines_pixel_values.append(pixel_values)
    #TRY END
    '''

    return contours_image, points_on_contour, lines_len, lines, lines_pixel_values

def rings_along_radial_lines(lines_pixel_values):
    radial_lines_info = []

    for pixel_values in lines_pixel_values:
        subseq_ = subseq(pixel_values, len(pixel_values))
        peaks_ = peaks(pixel_values)
        binary_ = binary(pixel_to_image(pixel_values)[0])
        moving_window_averages_ = moving_window_averages(pixel_values)

        radial_lines_info.append([subseq_, len(peaks_[0]), len(binary_["changes"]), len(moving_window_averages_["changes"])])

    rings_count = [[], [], [], []]

    for radial_line_info in radial_lines_info:
        rings_count[0].append((radial_line_info[0])//2)
        rings_count[1].append((radial_line_info[1])//2)
        rings_count[2].append((radial_line_info[2])//2)
        rings_count[3].append((radial_line_info[3])//2)

    return rings_count        

def visualize(model, gray_image, polar_image, edge_image, prediction, points_on_contour, lines, lines_pixel_values, rings_count, ranking, no_of_lines):
    plot_radial_lines(model, gray_image, polar_image, edge_image, prediction, points_on_contour)
    plot_lines(model, gray_image, polar_image, edge_image, lines)
    plot_change_along_radial_lines(lines_pixel_values)

    for idx in range(len(ALGOS)):
        name = ALGOS[idx]
        x = np.arange(start=1, stop=no_of_lines+1, step=1)
        y = rings_count[idx]
        plot_rings_along_lines(name, x, y, ranking)

def stats_min_max_mean(no_of_images, images_rings, rankings):
    min_max_mean = [[], [], [], []]

    for idx in range(no_of_images):
        rings_count = images_rings[idx]
        orig_mean = rankings[idx]

        min_max_mean[0].append([min(rings_count[0]), max(rings_count[0]), np.mean(rings_count[0]), orig_mean])
        min_max_mean[1].append([min(rings_count[1]), max(rings_count[1]), np.mean(rings_count[1]), orig_mean])
        min_max_mean[2].append([min(rings_count[2]), max(rings_count[2]), np.mean(rings_count[2]), orig_mean])
        min_max_mean[3].append([min(rings_count[3]), max(rings_count[3]), np.mean(rings_count[3]), orig_mean])
    
    return min_max_mean

def visualize_min_max_mean(no_of_images, min_max_mean):

    for idx in range(len(ALGOS)):
        name = ALGOS[idx]
        x = np.arange(start=1, stop=no_of_images + 1, step=1)
        y = np.array(min_max_mean[idx])
        plot_min_max_mean_orig(name, x, y)

def save_rings(no_of_images, model, no_of_lines, names, images_rings, min_max_mean, rings_fname):

    rings_rows = []
    
    for idx in range(no_of_images):
        rings_count = images_rings[idx]
    
        rings_rows.append([model.value, ALGOS[0], names[idx]] + rings_count[0] + min_max_mean[0][idx])
        rings_rows.append([model.value, ALGOS[1], names[idx]] + rings_count[1] + min_max_mean[1][idx])
        rings_rows.append([model.value, ALGOS[2], names[idx]] + rings_count[2] + min_max_mean[2][idx])
        rings_rows.append([model.value, ALGOS[3], names[idx]] + rings_count[3] + min_max_mean[3][idx])
    
    # save rings
    lines_labels = []

    for idx in range(no_of_lines):
        lines_labels.append("line_" + str(idx+1))

    rings_header =  ["model", "algo", "image"] + lines_labels + ["min", "max", "mean", "orig"]
    df = pd.DataFrame(rings_rows, columns= rings_header)

    with open(rings_fname, 'a') as f:
        df.to_csv(f, header=f.tell()==0, index = False)

def stats_rmse(no_of_images, images_rings, rankings):
    Y_pred = [[], [], [], []]
    Y_true = []

    for idx in range(no_of_images):
        rings_count = images_rings[idx]
        orig_mean = rankings[idx]

        Y_pred[0].append(np.mean(rings_count[0]))
        Y_pred[1].append(np.mean(rings_count[1]))
        Y_pred[2].append(np.mean(rings_count[2]))
        Y_pred[3].append(np.mean(rings_count[3]))

        Y_true.append(orig_mean)

    mean_squared_error_arr = [0] * len(ALGOS)
    mean_squared_error_arr[0] = mean_squared_error(Y_true, Y_pred[0], squared=False).round(3)
    mean_squared_error_arr[1] = mean_squared_error(Y_true, Y_pred[1], squared=False).round(3)
    mean_squared_error_arr[2] = mean_squared_error(Y_true, Y_pred[2], squared=False).round(3)
    mean_squared_error_arr[3] = mean_squared_error(Y_true, Y_pred[3], squared=False).round(3)

    return mean_squared_error_arr

def visualize_rmse(mean_squared_error_arr):
    x = np.array(ALGOS)
    y = np.array(mean_squared_error_arr)
    plot_rmse(x, y)

def save_rmse(model, mean_squared_error_arr, rmse_fname):
    stats_header =  ["model"] + ALGOS
    stats_rows = [model.value] + mean_squared_error_arr
    stats_row_data = [stats_rows]
    df = pd.DataFrame(stats_row_data, columns= stats_header)

    with open(rmse_fname, 'a') as f:
        df.to_csv(f, header=f.tell()==0, index = False)
        display(df)