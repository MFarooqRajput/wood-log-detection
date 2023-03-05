from pkg.pipeline import prepare_dataset
from pkg.pipeline import get_prediction_ranking
from pkg.pipeline import process_image
from pkg.pipeline import change_along_radial_lines
from pkg.pipeline import info_along_radial_lines
from pkg.pipeline import visualize
from pkg.pipeline import stats

def with_model(model, no_of_lines, no_of_points, data_dir, ranking_url, pickle_url, rings_url, mse_url):

    names = []
    rankings = []
    images_rings = []

    ## prepare dataset
    images, images_ranking_prediction_df = prepare_dataset(data_dir, ranking_url, pickle_url)

    for idx, image in enumerate(images):

        ## get prediction ranking
        name, prediction, ranking = get_prediction_ranking(idx, images_ranking_prediction_df)
        print("%s %s %s" % (idx, name, ranking))
        
        ## process image
        gray_image, polar_image, edge_image = process_image(model, prediction, image)

        ## change along radial lines
        contours_image, points_on_contour, lines_len, lines, lines_pixel_values = change_along_radial_lines(model, gray_image, polar_image, edge_image, prediction, no_of_lines, no_of_points)

        ## info along radial lines
        rings_count = info_along_radial_lines(lines_pixel_values)

        ## visualize
        visualize(model, gray_image, polar_image, edge_image, prediction, points_on_contour, lines, lines_pixel_values, rings_count, ranking, no_of_lines)
        
        names.append(name)
        rankings.append(ranking)
        images_rings.append(rings_count)

    ## stats
    stats(len(images), model, no_of_lines, names, rankings, images_rings, rings_url, mse_url)