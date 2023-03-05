import pandas as pd
import tensorflow as tf
import cv2
import numpy as np

from pkg.helper import append_dict_to_df

def pith_prediction(image):
    model = tf.keras.models.load_model('PithDetector')
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    
    return prediction

def get_pith_prediction(images):
    prediction_df = pd.DataFrame(columns = ['image_index', 'prediction'])
    for idx, image in enumerate(images):
        prediction = pith_prediction(image)
        prediction_df = append_dict_to_df(prediction_df,{'image_index' : idx, 'prediction' : prediction})
    
    return prediction_df