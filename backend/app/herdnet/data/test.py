# encode image to base64
import base64

#print current path 
import os
print("Current Working Directory:", os.getcwd())

with open("./test_infer/S_07_05_16_DSC00570.JPG", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# print(encoded_string)


# save to file
with open("encoded_image.txt", "w") as file:
    file.write(encoded_string)


# perform request to a local server in port 8000 to send the image as part of the body
import requests

#ping request
url = "http://localhost:8000/ping"
response = requests.get(url)
print(response.json())

url = "http://localhost:8000/inference/infer"

response = requests.post(url, json={"image": encoded_string})

print(response.json())

# decode image from base64
decoded_image = base64.b64decode(response.json()["image"])

# save to file
with open("decoded_image.jpg", "wb") as file:
    file.write(decoded_image)
