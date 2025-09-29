import os
import json
import base64
import io
import requests
import cv2
import torch
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

# --- THIS IS THE FIX ---
# Set up proper logging to ensure messages are displayed in real-time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- INITIALIZE APP and SERVICES ---
app = Flask(__name__)
logging.info("Flask app initialized.")

# --- LOAD AI MODEL (Done once on container startup) ---
logging.info("Loading CLIP model and processor...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logging.info(f"CLIP model and processor loaded successfully onto device: {device}")
except Exception as e:
    logging.error(f"FATAL: Could not load CLIP model. Error: {e}")
    # In a real app, you might want to exit if the model fails to load
    model, processor, device = None, None, None

# --- HELPER FUNCTIONS ---
def get_image_from_url(url):
    """Downloads an image from a URL and returns a PIL Image."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logging.warning(f"Error getting image {url}: {e}")
        return None

# --- MAIN ANALYSIS FUNCTION ---
def verify_multimodal_consistency(media_url, text_content, processor, model, device):
    """Performs the multimodal consistency check using CLIP."""
    if not media_url:
        logging.info("No media URL provided. Skipping multimodal analysis.")
        return 0.0

    image = get_image_from_url(media_url)

    if image is None:
        logging.warning("Could not retrieve a valid image for analysis.")
        return 0.0

    try:
        # Give the model two choices: the actual text and a neutral/opposite description.
        # This forces a more meaningful probability score.
        inputs = processor(
            text=[text_content, "a generic, unrelated image"],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # The score is the probability of the first text description (the actual one)
        score = probs[0][0].item()
        logging.info(f"CLIP similarity score: {score:.4f}")
        return score
    except Exception as e:
        logging.error(f"Error during CLIP model inference: {e}")
        return 0.0

# --- FLASK ROUTE ---
@app.route("/analyze", methods=["POST"])
def analyze():
    raw_data = request.get_json()
    item_id = raw_data.get("id", "unknown")
    logging.info(f"Analyzer received item: {item_id}")

    if not model or not processor:
        logging.error("Cannot process request because CLIP model is not loaded.")
        return jsonify({"error": "Model not loaded"}), 500

    text_for_clip = raw_data.get("text") or f"{raw_data.get('title', '')} {raw_data.get('description', '')}"

    consistency_score = verify_multimodal_consistency(
        raw_data.get("media_url"),
        text_for_clip,
        processor, model, device
    )
    
    # Placeholder for DistilBERT claim detection
    is_claim = "breaking" in text_for_clip.lower()
    
    enriched_data = {
        "source_data": raw_data,
        "analysis_results": {
            "is_potential_claim": is_claim,
            "multimodal_consistency_score": consistency_score
        }
    }
    
    return jsonify(enriched_data)

if __name__ == "__main__":
    logging.info("Multimodal Analyzer Service starting...")
    serve(app, host="0.0.0.0", port=5001)

