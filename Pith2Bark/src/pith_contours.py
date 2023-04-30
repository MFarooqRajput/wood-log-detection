import matplotlib.pyplot as plt

from src.cv_helper import convert_to_polar
from src.cv_helper import convert_to_polar_pith
from src.cv_helper import get_contours
from src.cv_helper import get_points_on_contour
from src.cv_helper import convert_to_polar
from src.cv_helper import convert_to_polar_pith

from src.cv_model import Model

from src.cv_pipeline import prepare_dataset
from src.cv_pipeline import get_prediction_ranking
from src.cv_pipeline import process_image

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