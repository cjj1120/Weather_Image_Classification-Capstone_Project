# Test the docker image built 
import requests

url = 'https://eowiay90ba.execute-api.us-east-2.amazonaws.com/test/predict'
data = {'url' : 'https://raw.githubusercontent.com/cjj1120/Weather_Image_Classification-Capstone_Project/main/Data/Add/Custom-test-img/rainbow-test.jpg'}

result = requests.post(url, json=data).json()
print(result) 


