import pandas as pd

from pkg.visualize import plot_multi_mse
from pkg.visualize import plot_images, plot_images_group
from pkg.visualize import plot_image_with_pith, plot_gray_polar_pith_images, plot_contours_with_points
from pkg.visualize import plot_rings_along_lines, plot_min_max_mean_orig
from pkg.helper import *
from pkg.model import *
from pkg.pith_prediction import pith_prediction

from pkg.pipeline import prepare_dataset
from pkg.pipeline import get_prediction_ranking
from pkg.pipeline import process_image

def multi_mse(mse_fname, show_group=True):
    df = pd.read_csv(mse_fname, index_col='model')
    col_list = list(df)
    min_mse = df[col_list].min(axis=1).min()
    max_mse = df[col_list].max(axis=1).max()
    winner_idx = df[col_list].min(axis=1).idxmin()
    no_of_bins = max_mse // 50

    print("%s %s (%s) \n" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin()))
    display(df)
    print("Min: %s, Max: %s" % (min_mse, max_mse))

    plot_multi_mse(df, min_mse, no_of_bins)
    df_arr = []

    if show_group:
        df_arr.append(df.iloc[:15,:])
        df_arr.append(df.iloc[15:27,:])
        df_arr.append(df.iloc[27:,:])

    for i in range(len(df_arr)):
        df = df_arr[i]
        col_list = list(df)
        min_mse = df[col_list].min(axis=1).min()
        max_mse = df[col_list].max(axis=1).max()
        no_of_bins = max_mse // 50

        display(df)
        print("Min: %s, Max: %s" % (min_mse, max_mse))
        plot_multi_mse(df, min_mse, no_of_bins)

def convert_image(data_dir, idx):

    images, images_df = read_images(data_dir)
    image = images[idx]
    prediction = pith_prediction(image)

    converted_images = []

    gray_image = convert_to_grayscale(image)
    converted_images.append(gray_image)
    polar_image = convert_to_polar(gray_image)
    converted_images.append(polar_image)
    polar_image = convert_to_polar_pith(gray_image, prediction)
    converted_images.append(polar_image)
        
    sobel_edge_image = convert_to_sobel_edge(gray_image)
    converted_images.append(sobel_edge_image)
    sobel_edge_image = convert_to_sobel_edge_blur(gray_image)
    converted_images.append(sobel_edge_image)
    canny_edge_image = convert_to_canny_edge(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_otsu(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_triangle(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_manual(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_otsu(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_triangle(gray_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_manual(gray_image)
    converted_images.append(canny_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge(gray_image)
    converted_images.append(laplacian_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge_blur(gray_image)
    converted_images.append(laplacian_edge_image)

    polar_image = convert_to_polar(gray_image)

    sobel_edge_image = convert_to_sobel_edge(polar_image)
    converted_images.append(sobel_edge_image)
    sobel_edge_image = convert_to_sobel_edge_blur(polar_image)
    converted_images.append(sobel_edge_image)
    canny_edge_image = convert_to_canny_edge(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_otsu(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_triangle(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_manual(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_otsu(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_triangle(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_manual(polar_image)
    converted_images.append(canny_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge(polar_image)
    converted_images.append(laplacian_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge_blur(polar_image)
    converted_images.append(laplacian_edge_image)

    polar_image = convert_to_polar_pith(gray_image, prediction)

    sobel_edge_image = convert_to_sobel_edge(polar_image)
    converted_images.append(sobel_edge_image)
    sobel_edge_image = convert_to_sobel_edge_blur(polar_image)
    converted_images.append(sobel_edge_image)
    canny_edge_image = convert_to_canny_edge(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_otsu(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_triangle(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_manual(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_otsu(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_triangle(polar_image)
    converted_images.append(canny_edge_image)
    canny_edge_image = convert_to_canny_edge_blur_manual(polar_image)
    converted_images.append(canny_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge(polar_image)
    converted_images.append(laplacian_edge_image)
    laplacian_edge_image = convert_to_laplacian_edge_blur(polar_image)
    converted_images.append(laplacian_edge_image)

    model_titles = ["Grayscale", "Polar", "Polar Pith", 
                    "Sobel", "Sobel(blur)", "Canny", "Canny(otsu)", "Canny(triangle)", "Canny(manual)", "Canny(blur)", "Canny(blur otsu)", "Canny(blur triangle)", "Canny(blur manual)", "Laplacian", "Laplacian(blur)", 
                    "Polar Sobel", "Polar Sobel(blur)", "Polar Canny", "Polar Canny(otsu)", "Polar Canny(triangle)", "Polar Canny(manual)", "Polar Canny(blur)", "Polar Canny(blur otsu)", "Polar Canny(blur triangle)", "Polar Canny(blur manual)", "Polar Laplacian", "Polar Laplacian(blur)", 
                    "Polar Pith Sobel", "Polar Pith Sobel(blur)", "Polar Pith Canny", "Polar Pith Canny(otsu)", "Polar Pith Canny(riangle)", "Polar Pith Canny(manual)", "Polar Pith Canny(blur)", "Polar Pith Canny(blur otsu)", "Polar Pith Canny(blur triangle)", "Polar Pith Canny(blur manual)", "Polar Pith Laplacian", "Polar Pith Laplacian(blur)"]

    return model_titles, converted_images

def converted_image(model_titles, converted_images):
    plot_images(model_titles, converted_images)

def converted_image_group_one(model_titles, converted_images):
    titles = []
    images = []

    for i in range(15):
        image = converted_images[i]
        images.append(image)
        titles.append(model_titles[i])

    plot_images_group(titles, images, 5, 3)

def converted_image_group_two(model_titles, converted_images):
    titles = []
    images = []

    for i in range(12):
        images.append(converted_images[i+15])
        titles.append(model_titles[i+15])

    plot_images_group(titles, images, 4, 3)

def converted_image_group_three(model_titles, converted_images):
    titles = []
    images = []

    for i in range(12):
        images.append(converted_images[i+27])
        titles.append(model_titles[i+27])

    plot_images_group(titles, images, 4, 3)

def prepare_data(model, data_dir, ranking_fname, pickle_fname):
    grayscale_images = []
    predictions = []
    images, images_ranking_prediction_df = prepare_dataset(data_dir, ranking_fname, pickle_fname)

    for idx, image in enumerate(images):
        name, prediction, ranking = get_prediction_ranking(idx, images_ranking_prediction_df)
        print("%s %s %s" % (idx, name, ranking))
        gray_image, polar_image, edge_image = process_image(model, prediction, image)
        grayscale_images.append(gray_image)
        predictions.append(prediction)

    return images, grayscale_images, predictions

def images_with_pith(images, predictions):

    for idx, image in enumerate(images):
        prediction = predictions[idx]
        plot_image_with_pith(image, prediction)

def gray_polar_pith_images(grayscale_images, predictions):
    plot_gray_polar_pith_images(grayscale_images, predictions)

def contours_with_points(model, grayscale_images, predictions, no_of_lines=32):

    for idx, gray_image in enumerate(grayscale_images):
        prediction = predictions[idx]

        if model == Model.GRAYSCALE:
            source_image = gray_image
        elif model == Model.POLAR:
            source_image = convert_to_polar(gray_image)
        elif model == Model.POLAR_PITH:
            source_image = convert_to_polar_pith(gray_image, prediction)
        else:
            pass

        contours_image, contours_array = get_contours(source_image)
        points_on_contour = get_points_on_contour(contours_array, no_of_lines)

        plot_contours_with_points(contours_image, points_on_contour)

def winner(mse_url):
    df = pd.read_csv(mse_url, index_col='model')
    col_list = list(df)
    winner_idx = df[col_list].min(axis=1).idxmin()

    print("%s %s (%s)" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin())) 
    #print(df.loc[winner_idx])
    
    return winner_idx, df.loc[winner_idx].idxmin()

def rings(rings_url, winner_idx, algo):
    df = pd.read_csv(rings_url, index_col='model') 
    sub_df = df.loc[winner_idx:winner_idx]
    rings_df = sub_df.loc[sub_df['algo'] == algo]
    #display(rings_df)
    return rings_df

def winner_rings(mse_url, rings_url):
    model, algo = winner(mse_url)
    df = rings(rings_url, model, algo)

    return df

def winner_visual(df):
    lines_cols = [col for col in df.columns if 'line' in col]
    lines = [s.strip('line_') for s in lines_cols]
    min_max_mean = []
    algo_name = None
    no_of_images = 0

    for index, row in df.iterrows():
        name = row['algo']
        x = lines
        y = row[lines_cols].values
        ranking = row['orig']
        plot_rings_along_lines(name, x, y, ranking)

        min_max_mean.append([row['min'], row['max'], row['mean'], row['orig']])

        if algo_name is None:
            algo_name = name

        no_of_images += 1

    x = np.arange(start=1, stop=no_of_images + 1, step=1)
    y = np.array(min_max_mean)
    plot_min_max_mean_orig(algo_name, x, y)

def check_correctness(df, original_label, predicted_label):
    correct = len(df[(df[original_label] == df[predicted_label])])
    incorrect = len(df) - correct

    return (correct,incorrect,(correct*100/(correct+incorrect)))

def damage(damage_url, df_1, df_2, df_3, df_4):
    df_damage = pd.read_excel(damage_url)

    #print(damage_df['Image'].count())
    #print(df_ringsranking_sixteen['image'].count())
    #print(df_pine_sixteen['image'].count())
    #print(df_fur_sixteen['image'].count())
    #print(df_tracy_sixteen['image'].count())

    #append all DataFrames into one DataFrame
    df_concat = pd.concat([df_1, df_2, df_3, df_4])

    #merge with damage
    df = pd.merge(left=df_concat, right=df_damage, left_on="image", right_on="Image")

    df.reset_index(drop=True, inplace=True)

    return df