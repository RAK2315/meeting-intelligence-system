from google import genai

client = genai.Client(api_key="AIzaSyCXA49dj9VPOg3lhqNfsW611E3ET-4d5aU")
for model in client.models.list():
    print(model.name)