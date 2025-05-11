import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image

# -------------------------------
# 1. Load Image-to-Text Model
# -------------------------------
print("Loading Image-to-Text Model...")
torch.cuda.empty_cache()
vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def extract_text_from_image(image_path):
    """Extracts text from an image using Qwen2-VL."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = vision_model.generate(**inputs)
    extracted_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return extracted_text

# -------------------------------
# 2. Load Sentiment Analysis Model
# -------------------------------
print("Loading Sentiment Analysis Model...")
sentiment_model = load_model("sentiment_model.h5")

# Load tokenizer (must match the one used in training)
tokenizer = Tokenizer(num_words=10000)

def preprocess_text(text, tokenizer, max_length=500):
    """Tokenizes and pads text to match training format"""
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length)
    return padded

def predict_sentiment(text):
    """Predicts the sentiment of given text using the loaded model"""
    processed_text = preprocess_text(text, tokenizer)
    prediction = sentiment_model.predict(processed_text)
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    return sentiment

# -------------------------------
# 3. Run Pipeline on an Image
# -------------------------------
if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with actual image path
    print(f"Processing image: {image_path}")

    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted Text: {extracted_text}")

    sentiment = predict_sentiment(extracted_text)
    print(f"Predicted Sentiment: {sentiment}")
