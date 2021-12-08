from keras_image_helper import create_preprocessor
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np 


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    return img


interpreter =  tflite.Interpreter(model_path='weather-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes=['fogsmog',
         'hail',
         'lightning',
         'rain',
         'rainbow',
         'rime',
         'sandstorm',
         'snow']


#test URL
url ='https://raw.githubusercontent.com/cjj1120/Weather_Image_Classification-Capstone_Project/main/Data/Add/Custom-test-img/rainbow-test.jpg'

def predict_url(url=''):
    img = download_image(url)
    img = prepare_image(img)
    x = np.array(img,dtype=np.float32)
    X = np.array([x])
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    interpreter.get_tensor(output_index)
    pred = interpreter.get_tensor(output_index)
    dict_pred = dict(zip(classes, pred[0]))
    return dict_pred

def lambda_handler(event, context):
    url = event['url']
    result = predict_url(url)
    return result 