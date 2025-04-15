import os
import logging
import google.generativeai as genai
from PIL import Image
import base64
import io
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables, Gemini features may not work properly")

genai.configure(api_key=API_KEY)
chat_history = {}

def get_treatment_recommendation(disease_name):
    try:
        disease = disease_name.replace('_', ' ')
        prompt = f"""
        You are a plant disease expert. Provide treatment recommendations for plants affected by {disease}.
        1. Brief description of the disease
        2. Symptoms
        3. Treatment (organic + chemical)
        4. Prevention tips
        """
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text if response else "Unable to generate treatment."
    except Exception as e:
        logger.error(f"Treatment error: {str(e)}")
        return "Fallback: consult a local expert."

def classify_disease_with_image(image_path):
    try:
        if not API_KEY:
            return "API key missing", 0.0

        model = genai.GenerativeModel("gemini-1.5-pro-vision")
        image = Image.open(image_path).convert("RGB")

        prompt = (
            "You are an expert plant pathologist. Identify the disease in this plant image, "
            "and respond only with the disease name (e.g., 'Tomato Late Blight'). "
            "If it's healthy, say 'Healthy Plant'."
        )

        response = model.generate_content([prompt, image])
        prediction = response.text.strip()

        return prediction, 0.95  # Gemini doesn't return confidence, so we assign a default

    except Exception as e:
        logger.error(f"Gemini Vision classification error: {str(e)}")
        return "Unknown Disease", 0.0

def initialize_chat(session_id):
    try:
        if not API_KEY:
            return False

        model = genai.GenerativeModel('gemini-1.5-pro')
        chat = model.start_chat()
        chat.send_message("You are a plant health assistant.")
        chat_history[session_id] = chat
        return True
    except Exception as e:
        logger.error(f"Chat init error: {str(e)}")
        return False

def chat_with_gemini(session_id, user_message):
    try:
        if not API_KEY:
            return "API key missing."

        if session_id not in chat_history:
            if not initialize_chat(session_id):
                return "Failed to start chat session."

        chat = chat_history[session_id]
        response = chat.send_message(user_message)
        return response.text if response and hasattr(response, 'text') else "No response."
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return "Error occurred during chat."
