from huggingface_hub import hf_hub_download

model_path = hf_hub_download("srikanth-maganti/Retinitis-pigmentosa-detection", "head_model.h5")

import tensorflow as tf


def predict(image):
    
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(image)
    return prediction