import streamlit as st
import os
import pyttsx3
import pytesseract
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
import io
import re
import time
from google.api_core.exceptions import ResourceExhausted

# --- SETUP ---

# Configure API Keys
GEMINI_API_KEY = "AIzaSyD8aUaGFlsWBAKmiaNkCTW4WA3AIZiGKyU"  # Replace with your actual Gemini API Key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Set Tesseract OCR Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize AI Models
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Emoji Mapping for Captions (kept for future use)
emoji_dict = {
    "dog": "🐶", "cat": "🐱", "car": "🚗", "bike": "🚲",
    "sun": "☀️", "tree": "🌳", "beach": "🏖️", "mountain": "🏔️",
    "food": "🍕", "people": "👨‍👩‍👧‍👦", "city": "🌆", "sky": "🌌",
    "ocean": "🌊", "bird": "🐦", "flower": "🌸", "smile": "😊"
}

# --- FUNCTIONS ---

def extract_text_from_image(image):
    """Extracts text from an image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(image_data):
    """Generates a scene description using Google Gemini with retry logic."""
    retries = 5
    for attempt in range(retries):
        try:
            response = genai_model.generate_content(["Describe this image.", image_data])
            return response.text
        except ResourceExhausted as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Rate limit exceeded. Please try again later.")
                raise e

def generate_gpt_caption(image):
    """Generates a short one-line image caption using Google Gemini."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    image_data = {
        "mime_type": "image/png",
        "data": image_bytes.getvalue()
    }

    try:
        # Short prompt for fast, concise response
        response = genai_model.generate_content(["Give a one-line caption for this image.", image_data])
        caption = response.text.strip().split(".")[0] + "."
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Could not generate caption."

def add_emojis(caption):
    """Adds emojis to the caption based on detected keywords (optional use)."""
    for word, emoji in emoji_dict.items():
        if re.search(rf'\b{word}\b', caption, re.IGNORECASE):
            caption += f" {emoji}"
    return caption

def calculate_bleu(reference, generated):
    """Calculates BLEU score to evaluate caption accuracy."""
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    return sentence_bleu([reference_tokens], generated_tokens)

# --- STREAMLIT UI ---

st.set_page_config(page_title="Image Caption Generator", layout="wide", page_icon="🐶")

st.markdown("<h1 style='text-align: center; color: #0662f6;'>Image Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Empowering the Partially Sighted with AI</h4>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("C:\\Users\\samudrala sindhuja\\Downloads\\sidebar_image.jpg", width=250)
st.sidebar.title("About")
st.sidebar.markdown(
    """
    🐶 **Features**  
    - 🐶 AI-generated **scene descriptions**  
    - 🐶 **Image captions** with **emojis**  
    - 🐶 **Text extraction** from images  
    - 🐶 **Text-to-speech**  
    """
)

# File Uploader
uploaded_file = st.file_uploader("🐶 Upload an Image", type=["jpg", "jpeg", "png"])

# Initialize session state for extracted text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🐶 Uploaded Image", use_column_width=True)

    col1, col2, col3 = st.columns(3)

    if col1.button("🐶 Describe Scene"):
        with st.spinner("Analyzing scene..."):
            image_data = {"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}
            description = generate_scene_description(image_data)
            st.write(description)

    if col2.button("🐶 Generate Caption"):
        with st.spinner("Generating caption..."):
            eng_caption = generate_gpt_caption(image)
            st.write(f"**Caption:** {eng_caption}")

    if col3.button("🐶 Extract Text"):
        with st.spinner("Extracting text..."):
            st.session_state.extracted_text = extract_text_from_image(image)
            st.text_area("Extracted Text", st.session_state.extracted_text, height=150)

    if st.button("🐶 Listen to Text") and st.session_state.extracted_text:
        text_to_speech(st.session_state.extracted_text)

    reference_caption = st.text_input("🐶 Enter Reference Caption for BLEU Score:")
    if reference_caption:
        bleu_score = calculate_bleu(reference_caption, eng_caption)
        st.write(f"🐶 BLEU Score: {bleu_score:.2f}")

# Footer
st.markdown("<hr><footer style='text-align:center;'>Built with ❤️ by Samudrala Sindhuja</footer>", unsafe_allow_html=True)
