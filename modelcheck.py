from google import genai

client = genai.Client(api_key="AIzaSyCyI6FED0bSBN8Ibcu9BuAB3dkdVXLFvsc")

for model in client.models.list():
    print(model.name)