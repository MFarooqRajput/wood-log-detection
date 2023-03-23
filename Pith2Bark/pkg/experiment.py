from pkg.pipeline import prepare_dataset
from pkg.pipeline import get_prediction_ranking
from pkg.pipeline import process_image
from pkg.pipeline import change_along_radial_lines
from pkg.pipeline import rings_along_radial_lines
from pkg.pipeline import visualize
from pkg.pipeline import stats_min_max_mean, visualize_min_max_mean, save_rings
from pkg.pipeline import stats_mse, visualize_mse, save_mse

def with_model(model, no_of_lines, no_of_points, data_dir, ranking_fname, pickle_fname, rings_fname=None, mse_fname=None):

    names = []
    rankings = []
    images_rings = []

    ## prepare dataset
    images, images_ranking_prediction_df = prepare_dataset(data_dir, ranking_fname, pickle_fname)

    for idx, image in enumerate(images):

        ## get prediction ranking
        name, prediction, ranking = get_prediction_ranking(idx, images_ranking_prediction_df)
        print("%s %s %s" % (idx, name, ranking))
        
        ## process image
        gray_image, polar_image, edge_image = process_image(model, prediction, image)

        ## change along radial lines
        contours_image, points_on_contour, lines_len, lines, lines_pixel_values = change_along_radial_lines(model, gray_image, polar_image, edge_image, prediction, no_of_lines, no_of_points)

        ## rings along radial lines
        rings_count = rings_along_radial_lines(lines_pixel_values)

        ## visualize
        visualize(model, gray_image, polar_image, edge_image, prediction, points_on_contour, lines, lines_pixel_values, rings_count, ranking, no_of_lines)
        
        names.append(name)
        rankings.append(ranking)
        images_rings.append(rings_count)

    ## min max mean
    min_max_mean = stats_min_max_mean(len(images), images_rings, rankings)
    visualize_min_max_mean(len(images), min_max_mean)

    if rings_fname is not None:
        save_rings(len(images), model, no_of_lines, names, images_rings, min_max_mean, rings_fname)

    ## mse
    mean_squared_error_arr = stats_mse(len(images), images_rings, rankings)
    visualize_mse(mean_squared_error_arr)

    if mse_fname is not None:
        save_mse(model, mean_squared_error_arr, mse_fname)