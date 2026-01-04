from flask import Flask, request, jsonify
import cv2
import numpy as np
from utils.color_analysis import extract_dominant_color
from utils.recommendations import suggest_outfits
from PIL import Image
import io

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['image']
    img = Image.open(file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    dominant_color = extract_dominant_color(img_cv)
    avg_brightness = np.mean(img_cv)

    if avg_brightness > 170:
        skin_tone = "fair"
    elif avg_brightness > 100:
        skin_tone = "medium"
    else:
        skin_tone = "dark"

    recs = suggest_outfits(skin_tone, dominant_color)
    return jsonify({"skin_tone": skin_tone, "dominant_color": dominant_color, "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)
