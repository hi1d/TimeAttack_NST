import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow_hub as hub
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = os.path.join(BASE_DIR, 'images/result.png')


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_style(style_path, max_dim):

    img = tf.io.read_file(style_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 전체 이미지의 비율을 유지하면서, 원하는 크기로 변환
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def upload_tensor_img(tensor):
    tensor = np.array(tensor*255 ,dtype=np.uint8)
    image = Image.fromarray(tensor[0])
    buffer = BytesIO()
    image.save(RESULT_DIR, 'PNG')
    buffer.seek(0)
    return 

def nst_apply(url) :
    res = requests.get(url)
    img = Image.open(BytesIO(res.content)).convert('RGB') 
    content_image = tf.keras.preprocessing.image.img_to_array(img)    

    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (512, 512))
    style_path = tf.keras.utils.get_file('timeattack2.jpg', "https://images.velog.io/images/aopd48/post/d736f511-b8af-4950-857e-b99703b6235d/image.png")
    style_image = load_style(style_path, 256)
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img(stylized_image)
        
    return 

nst_apply("https://images.velog.io/images/aopd48/post/ef8387b2-e1ba-4d3d-812e-2178ff52fd88/11.jpeg")