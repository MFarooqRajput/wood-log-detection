from enum import Enum

class Model(Enum):
    GRAYSCALE = "Grayscale"
    POLAR = "Polar"
    POLAR_PITH = "Polar Pith"
    SOBEL_EDGE = "Sobel"
    SOBEL_EDGE_BLUR = "Sobel(blur)"
    CANNY_EDGE = "Canny"
    CANNY_EDGE_OTSU = "Canny(otsu)"
    CANNY_EDGE_TRIANGLE = "Canny(triangle)"
    CANNY_EDGE_MANUAL = "Canny(manual)"
    CANNY_EDGE_BLUR = "Canny(blur)"
    CANNY_EDGE_BLUR_OTSU = "Canny(blur otsu)"
    CANNY_EDGE_BLUR_TRIANGLE = "Canny(blur triangle)"
    CANNY_EDGE_BLUR_MANUAL = "Canny(blur manual)"
    LAPLACIAN_EDGE = "Laplacian"
    LAPLACIAN_EDGE_BLUR = "Laplacian(blur)"
    POLAR_SOBEL_EDGE = "Polar Sobel"
    POLAR_SOBEL_EDGE_BLUR = "Polar Sobel(blur)"
    POLAR_CANNY_EDGE = "Polar Canny"
    POLAR_CANNY_EDGE_OTSU = "Polar Canny(otsu)"
    POLAR_CANNY_EDGE_TRIANGLE = "Polar Canny(triangle)"
    POLAR_CANNY_EDGE_MANUAL = "Polar Canny(manual)"
    POLAR_CANNY_EDGE_BLUR = "Polar Canny(blur)"
    POLAR_CANNY_EDGE_BLUR_OTSU = "Polar Canny(blur otsu)"
    POLAR_CANNY_EDGE_BLUR_TRIANGLE = "Polar Canny(blur triangle)"
    POLAR_CANNY_EDGE_BLUR_MANUAL = "Polar Canny(blur manual)"
    POLAR_LAPLACIAN_EDGE = "Polar Laplacian"
    POLAR_LAPLACIAN_EDGE_BLUR = "Polar Laplacian(blur)"
    POLAR_PITH_SOBEL_EDGE = "Polar Pith Sobel"
    POLAR_PITH_SOBEL_EDGE_BLUR = "Polar Pith Sobel(blur)"
    POLAR_PITH_CANNY_EDGE = "Polar Pith Canny"
    POLAR_PITH_CANNY_EDGE_OTSU = "Polar Pith Canny(otsu)"
    POLAR_PITH_CANNY_EDGE_TRIANGLE = "Polar Pith Canny(triangle)"
    POLAR_PITH_CANNY_EDGE_MANUAL = "Polar Pith Canny(manual)"
    POLAR_PITH_CANNY_EDGE_BLUR = "Polar Pith Canny(blur)"
    POLAR_PITH_CANNY_EDGE_BLUR_OTSU = "Polar Pith Canny(blur otsu)"
    POLAR_PITH_CANNY_EDGE_BLUR_TRIANGLE = "Polar Pith Canny(blur triangle)"
    POLAR_PITH_CANNY_EDGE_BLUR_MANUAL = "Polar Pith Canny(blur manual)"
    POLAR_PITH_LAPLACIAN_EDGE = "Polar Pith Laplacian"
    POLAR_PITH_LAPLACIAN_EDGE_BLUR = "Polar Pith Laplacian(blur)"

def is_grayscale(model):
    if model == Model.GRAYSCALE:
        return True
    else:
        return False

def is_polar(model):
    if model == Model.POLAR or model == Model.POLAR_PITH:
        return True
    else:
        return False

def is_polar_edge(model):
    if model == Model.POLAR_SOBEL_EDGE or model == Model.POLAR_SOBEL_EDGE_BLUR or model == Model.POLAR_CANNY_EDGE or model == Model.POLAR_CANNY_EDGE_OTSU or model == Model.POLAR_CANNY_EDGE_TRIANGLE or model == Model.POLAR_CANNY_EDGE_MANUAL or model == Model.POLAR_CANNY_EDGE_BLUR or model == Model.POLAR_CANNY_EDGE_BLUR_OTSU or model == Model.POLAR_CANNY_EDGE_BLUR_TRIANGLE or model == Model.POLAR_CANNY_EDGE_BLUR_MANUAL or model == Model.POLAR_LAPLACIAN_EDGE or model == Model.POLAR_LAPLACIAN_EDGE_BLUR:
        return True
    else:
        return False

def is_polar_pith_edge(model):
    if model == Model.POLAR_PITH_SOBEL_EDGE or model == Model.POLAR_PITH_SOBEL_EDGE_BLUR or model == Model.POLAR_PITH_CANNY_EDGE or model == Model.POLAR_PITH_CANNY_EDGE_OTSU or model == Model.POLAR_PITH_CANNY_EDGE_TRIANGLE or model == Model.POLAR_PITH_CANNY_EDGE_MANUAL or model == Model.POLAR_PITH_CANNY_EDGE_BLUR or model == Model.POLAR_PITH_CANNY_EDGE_BLUR_OTSU or model == Model.POLAR_PITH_CANNY_EDGE_BLUR_TRIANGLE or model == Model.POLAR_PITH_CANNY_EDGE_BLUR_MANUAL or model == Model.POLAR_PITH_LAPLACIAN_EDGE or model == Model.POLAR_PITH_LAPLACIAN_EDGE_BLUR:
        return True
    else:
        return False