from src.helper import convert_to_polar
from src.helper import convert_to_polar_pith
from src.helper import get_contours
from src.helper import get_points_on_contour

from src.model import Model

from src.pipeline import prepare_dataset
from src.pipeline import get_prediction_ranking
from src.pipeline import process_image

from src.pith_contours_visual import plot_image_with_pith
from src.pith_contours_visual import plot_gray_polar_pith_images
from src.pith_contours_visual import plot_contours_with_points

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