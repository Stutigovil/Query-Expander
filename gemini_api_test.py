# import google.generativeai as genai

# API_KEY = ""

# try:
#     # Configure Gemini
#     genai.configure(api_key=API_KEY)

#     # Load model
#     model = genai.GenerativeModel("gemini-1.5-flash")

#     # Test prompt
#     response = model.generate_content("Say hello in one sentence.")

#     print("✅ API Key is working!")
#     print("Response:", response.text)

# except Exception as e:
#     print("❌ API Key is NOT working")
#     print("Error:", str(e))
import google.generativeai as genai

API_KEY = "YOUR API KEY"

try:
    # Configure Gemini
    genai.configure(api_key=API_KEY)

    print("\nChecking available Gemini models...\n")

    # List available models
    models = genai.list_models()

    found = False

    for model in models:
        print(model.name)
        found = True

    if not found:
        print("❌ No Gemini models accessible.")

    print("\nTesting generation...\n")

    # Load model
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate response
    response = model.generate_content(
        "Reply with: Gemini API working successfully"
    )

    print("✅ SUCCESS")
    print(response.text)

except Exception as e:
    print("\n❌ ERROR DETECTED\n")
    print(type(e).__name__)
    print(str(e))