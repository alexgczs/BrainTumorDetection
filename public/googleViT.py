import requests
import json

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer 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"}

def query(image):
    filename="data/test.jpeg"
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    print(response.content)
    return json.loads(response.content.decode("utf-8"))

output = query("cats.jpg")