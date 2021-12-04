# Project Title 
Weather classification 

## About the project 
Classification context, capture plant or just open space/ street view/ skyview image ?

## THink about 
certian plan close up shots (Micro) are meant to extract important features, meaning to generalize to real image, it has to be able to pick up the 
features of real image, so we kinda want to emphasize on the important features right? (so we want the opposite of regularization, optimization?)

Dropping few classes to redudce training time, and I thoguht they are zoomed-in shots on plant, as compare to other zoom-out shots, i want the dataset to be more consistent. 
Classes dropped `[dew, frost, glaze]`

## To reproduce 
1. First Build the docker image 
2. Run the docker image and map the port ( docker run -it --rm -p 9696:9696 zoomcamp-test )
3. Open terminal and enter this code (python predict-test.py)
4. You should be able to see the prediction. 



### Dataset 
https://www.kaggle.com/jehanbhathena/weather-dataset
There are total of 6862 images in this dataset. 
The dataset contain of 11 different calsses of weather:  dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow.

Classes included `[fog/smog, hail, lightning , rain, rainbow, rime, sandstorm, snow]`

## Model selection and explanation 
EfficientNet 
https://arxiv.org/pdf/1905.11946.pdf 
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet?utm_source=catalyzex.com