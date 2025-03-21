import os
import google.generativeai as genai

# Get the API key from environment variables
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("GEMINI_API_KEY not found in environment variables")
    exit(1)

# Configure the Gemini API client
genai.configure(api_key=API_KEY)

# List all available models
print("Listing available models:")
for model in genai.list_models():
    print(f"- {model.name}: {model.display_name}")
    print(f"  Supported generation methods: {model.supported_generation_methods}")
    print()