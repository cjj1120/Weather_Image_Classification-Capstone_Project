# Project Title: Weather classification 

## About the project 
This project aims to build a neural network to classify weather of 8 different classes. Transfer learning is used in this study. 

## Dataset 
https://www.kaggle.com/jehanbhathena/weather-dataset
There are total of 6862 images in this dataset. 
The dataset contain of 11 different calsses of weather:  dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow. <br>
3 classes were dropped (`[dew, frost, glaze]`) as they are zoom-in shots of plants, the other are general weather shots (sky, city, open space view). I thought this way, the dataset can be more consistent. <br>
8 Classes included in this project: `[fog/smog, hail, lightning , rain, rainbow, rime, sandstorm, snow]`

## To reproduce 
1. First Build the docker image 
2. Run the docker image and map the port ( docker run -it --rm -p 9696:9696 zoomcamp-test )
3. Open terminal and enter this code (python predict-test.py)
4. You should be able to see the prediction. 


## Model selection and explanation 
EfficientNet seems to give the best validation accuracy, we can see that Efficientnet perform well as compared to some of the other networks. Among all the Efficien Nets, EfficientNetB4 is choosen as it provides a good trade off of accuracy and speed. 
![Network](Asset/model-comparison2.png)
https://arxiv.org/pdf/1905.11946.pdf 
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet?utm_source=catalyzex.com


# Testing/ Running the project 
 
## To test the model locally (without Docker): 
```
Open console, Ipython 
import lambda_func
url ='https://raw.githubusercontent.com/cjj1120/Weather_Image_Classification-Capstone_Project/main/Data/Add/Custom-test-img/rainbow-test.jpg'
lambda_func.predict_url(url)
```


## Test with Docker locally:

1. Build docker image `docker build -t weather-model .  
2. Run the image `docker run -it --rm -p 8080:8080 weather-model:latest`
3. Test with following command `python test.py` 

## Test the lambda function with AWS API: 
In console, run `python test-AWS-API.py`