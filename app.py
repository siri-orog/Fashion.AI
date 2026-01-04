import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from utils.color_analysis import extract_dominant_color
from utils.tryon import try_on_user_face  # new try-on module

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Fashion Stylist 👗",
    page_icon="👗",
    layout="wide",
)

st.title("👗 AI Fashion Stylist & Virtual Try-On")
st.markdown("### Discover what outfits suit you best — upload or capture your photo!")

# ----------------------------
# IMAGE INPUT
# ----------------------------
option = st.radio("Choose image input:", ["📸 Webcam", "📁 Upload"], horizontal=True)

img_cv = None
if option == "📸 Webcam":
    img_data = st.camera_input("Capture your photo")
    if img_data:
        img = Image.open(img_data).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
elif option == "📁 Upload":
    file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

if img_cv is not None:
    st.image(img, caption="Your Photo", use_column_width=True)
else:
    st.info("Please upload or capture an image to continue.")
    st.stop()

# ----------------------------
# COLOR ANALYSIS
# ----------------------------
st.markdown("---")
st.subheader("🎨 Analyzing your dominant color...")

dominant_color = extract_dominant_color(img_cv)
r, g, b = dominant_color
st.color_picker("Detected dominant color", value=f'#{r:02x}{g:02x}{b:02x}', disabled=True)

# ----------------------------
# OUTFIT RECOMMENDATION LOGIC
# ----------------------------
st.markdown("---")
st.subheader("👕 Outfit Recommendations")

def get_recommendations(color):
    r, g, b = color
    if r > g and r > b:
        return [
            "Earthy tones like olive, mustard, and maroon look great.",
            "Fusion styles — like kurtis with jeans or Indo-western outfits — work beautifully.",
            "Experiment with prints and textured fabrics."
        ], "Fusion Wear"
    elif g > r and g > b:
        return [
            "Cool colors like blue, teal, and lavender are perfect.",
            "Try western wear — jeans, t-shirts, and jackets.",
            "Monochrome outfits with pastel accents look stylish."
        ], "Western Wear"
    elif b > r and b > g:
        return [
            "Bright colors like coral, yellow, and white enhance your look.",
            "Traditional outfits like sarees and lehengas suit you well.",
            "Opt for gold or silver jewelry to complement your outfit."
        ], "Traditional Wear"
    else:
        return [
            "Neutral shades like beige, grey, and navy blue are timeless.",
            "Business casual or modern formal wear fits you perfectly.",
            "Try minimalistic designs and clean silhouettes."
        ], "Formal / Modern Look"

recommendations, style = get_recommendations(dominant_color)

st.write(f"**Recommended style:** {style}")
for r in recommendations:
    st.markdown(f"- {r}")

# ----------------------------
# STYLE IMAGES (EXAMPLES)
# ----------------------------
st.markdown("---")
st.subheader("👗 Style Inspiration")

style_images = {
    "Fusion Wear": [
        "https://i.imgur.com/3z7VQZq.jpg",
        "https://i.imgur.com/3z4tPpX.jpg"
    ],
    "Western Wear": [
        "https://i.imgur.com/xD4NnKk.jpg",
        "https://i.imgur.com/hkdOqGq.jpg"
    ],
    "Traditional Wear": [
        "https://i.imgur.com/XzQhPLn.jpg",
        "https://i.imgur.com/AVaWcPb.jpg"
    ],
    "Formal / Modern Look": [
        "https://i.imgur.com/zG5vRrV.jpg",
        "https://i.imgur.com/XqPwA6x.jpg"
    ]
}

cols = st.columns(2)
if style in style_images:
    for i, img_url in enumerate(style_images[style][:2]):
        try:
            with cols[i]:
                st.image(img_url, use_column_width=True, caption=f"{style} example")
        except Exception:
            st.warning("Image not available.")
else:
    st.info("No images found for this style.")

# ----------------------------
# TRY-ON PREVIEW
# ----------------------------
st.markdown("---")
st.subheader("🎯 Try-On Preview (Virtual Face Swap)")

if style in style_images:
    imgs = style_images[style]
    pics_cols = st.columns(len(imgs))
    thumb_choice = None
    for i, url in enumerate(imgs):
        try:
            with pics_cols[i]:
                st.image(url, use_column_width=True, caption=f"{style} example", output_format="auto")
                if st.button(f"Try {i+1}", key=f"trybtn_{i}"):
                    thumb_choice = url
        except Exception:
            pics_cols[i].warning("Image not available")

    if thumb_choice is None:
        thumb_choice = st.selectbox("Or choose an example for try-on", imgs)

    if st.button("Generate Try-on Preview"):
        try:
            resp = requests.get(thumb_choice, timeout=10)
            model_pil = Image.open(BytesIO(resp.content)).convert("RGB")
            model_bgr = cv2.cvtColor(np.array(model_pil), cv2.COLOR_RGB2BGR)

            result_img, err = try_on_user_face(img_cv, model_bgr)
            if err:
                st.warning(err)
            else:
                st.subheader("👚 Try-On Result")
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                _, encoded_img = cv2.imencode('.jpg', result_img)
                st.download_button(
                    "Download Try-on Image",
                    encoded_img.tobytes(),
                    file_name="tryon_result.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"Try-on failed: {e}")
else:
    st.info("No style images available for try-on.")
