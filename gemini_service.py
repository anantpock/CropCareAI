import os
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables, Gemini features may not work properly")

# Configure the Gemini API client
genai.configure(api_key=API_KEY)

def get_treatment_recommendation(disease_name):
    """
    Get treatment recommendations for a plant disease using Google Gemini
    
    Args:
        disease_name (str): The name of the plant disease
    
    Returns:
        str: Treatment recommendations
    """
    try:
        # Clean up disease name (replace underscores with spaces)
        disease = disease_name.replace('_', ' ')
        
        # Prepare the prompt for Gemini
        prompt = f"""
        You are a plant disease expert. Provide treatment recommendations for plants affected by {disease}.
        
        Follow this structure in your response:
        1. Brief description of the disease
        2. Symptoms
        3. Treatment recommendations (organic and chemical options)
        4. Prevention tips
        
        Keep your response informative but concise (less than 500 words).
        """
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract and return the text
        if response:
            return response.text
        else:
            return "Unable to generate treatment recommendations at this time."
    
    except Exception as e:
        logger.error(f"Error generating treatment recommendations: {str(e)}")
        
        # Fallback response if Gemini API fails
        return f"""
        # Treatment Recommendations for {disease_name.replace('_', ' ')}

        ## Description
        This is a common plant disease that affects crops and ornamental plants.

        ## Symptoms
        - Discoloration of leaves
        - Spots or lesions
        - Wilting or stunted growth

        ## Treatment
        - Remove affected plant parts
        - Apply appropriate fungicide or insecticide
        - Ensure proper plant nutrition

        ## Prevention
        - Rotate crops
        - Use disease-resistant varieties
        - Maintain good air circulation
        - Water at the base of plants to keep foliage dry

        *Note: These are general recommendations. For specific treatment, please consult with a local agricultural extension service.*
        """
