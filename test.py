# Test the docker image built 
import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url' : 'https://raw.githubusercontent.com/cjj1120/Weather_Image_Classification-Capstone_Project/main/Data/Add/Custom-test-img/rainbow-test.jpg'}

result = requests.post(url, json=data).json()
print(result) 


