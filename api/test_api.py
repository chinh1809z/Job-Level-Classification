import requests


data = {
    "title": "Senior Software Engineer",
    "location": "Berlin",
    "description": "We are looking for a senior software engineer to develop and maintain software applications.",
    "function": "IT",
    "industry": "Information Technology"
}


response = requests.post(
    "http://127.0.0.1:5000/api/predict",
    json=data
)


print(response.status_code)
print(response.json())