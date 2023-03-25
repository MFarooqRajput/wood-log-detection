from src.helper import read_images
from src.helper import convert_to_grayscale
from src.helper import convert_to_polar
from src.helper import convert_to_polar_pith
from src.helper import convert_to_sobel_edge
from src.helper import convert_to_sobel_edge_blur
from src.helper import convert_to_canny_edge
from src.helper import convert_to_canny_edge_otsu
from src.helper import convert_to_canny_edge_triangle
from src.helper import convert_to_canny_edge_manual
from src.helper import convert_to_canny_edge_blur
from src.helper import convert_to_canny_edge_blur_otsu
from src.helper import convert_to_canny_edge_blur_triangle
from src.helper import convert_to_canny_edge_blur_manual
from src.helper import convert_to_laplacian_edge
from src.helper import convert_to_laplacian_edge_blur

from src.pith_prediction import pith_prediction

from src.process_image_visual import plot_images
from src.process_image_visual import plot_images_group

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

def converted_image_group(model_titles, converted_images, group):
    titles = []
    images = []

    if group == 1:
        iter = 15
        rows = 5
        cols = 3
        inc = 0
    elif group == 2:
        iter = 12
        rows = 4
        cols = 3
        inc = 15
    elif group == 3:
        iter = 12
        rows = 4
        cols = 3
        inc = 27
    else:
            pass

    for i in range(iter):
        image = converted_images[i+inc]
        images.append(image)
        titles.append(model_titles[i+inc])

    plot_images_group(titles, images, rows, cols)