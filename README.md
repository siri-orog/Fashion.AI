# Fashion.AI

**Fashion.AI** is an AI-powered application that provides personalized clothing recommendations and virtual try-on functionality. This project leverages machine learning and computer vision techniques to analyze clothing items, user preferences, and pose information to deliver intelligent styling suggestions.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Purpose](#purpose)  
3. [Project Development](#project-development)  
4. [Tools & Libraries](#tools--libraries)  
5. [Data Collection](#data-collection)  
6. [Data Preprocessing](#data-preprocessing)  
7. [Machine Learning Algorithms](#machine-learning-algorithms)  
8. [Challenges & Difficulties](#challenges--difficulties)  
9. [How to Run](#how-to-run)  
10. [How to Update](#how-to-update)  
11. [Future Improvements](#future-improvements)  

---

## Project Overview

Fashion.AI combines computer vision and machine learning to deliver:

- **Personalized styling recommendations** based on clothing analysis.  
- **Virtual try-on feature** allowing users to visualize outfits on themselves.  
- **Pose-aware overlay** to accurately place clothes on images.  

This project demonstrates practical AI application in fashion technology, useful for e-commerce, virtual styling apps, and fashion analytics.

---

## Purpose

The main objectives of Fashion.AI are:

- To help users discover personalized outfit combinations.  
- To automate styling recommendations using AI.  
- To provide an interactive virtual try-on experience.  
- To demonstrate end-to-end ML project development and deployment.

---

## Project Development

Fashion.AI was developed following these steps:

1. **Requirement analysis** – defined project scope and functionalities.  
2. **Data gathering** – collected fashion images and metadata.  
3. **Preprocessing** – cleaned images, normalized data, and extracted features.  
4. **ML model training** – implemented recommendation and overlay models.  
5. **Backend integration** – developed `stylist_backend.py` for processing requests.  
6. **Frontend integration** – created `app.py` as the main Flask interface.  
7. **Testing & Deployment** – tested virtual try-on and recommendation accuracy.  

---

## Tools & Libraries

The following tools and libraries were used:

- **Python 3.11** – programming language.  
- **Flask** – for building the web application (`app.py`).  
- **OpenCV** – for image processing and overlays.  
- **NumPy & Pandas** – data manipulation and feature extraction.  
- **TensorFlow / PyTorch** – deep learning frameworks for ML models.  
- **Matplotlib & Seaborn** – visualization (for testing and debugging).  
- **VS Code** – development IDE.  

---

## Data Collection

- Fashion images were collected from **open-source datasets** and e-commerce sites.  
- Metadata included **clothing type, color, style, and size**.  
- Data was **anonymized** and cleaned to ensure privacy compliance.

---

## Data Preprocessing

Key preprocessing steps:

1. **Image resizing** to a uniform size.  
2. **Normalization** of pixel values.  
3. **Feature extraction**: color histograms, keypoints, and pose landmarks.  
4. **Augmentation**: rotation, flipping, and scaling to increase dataset variability.  

---

## Machine Learning Algorithms

- **Recommendation Engine:**  
  - Algorithm: **K-Nearest Neighbors (KNN) / Collaborative Filtering**  
  - Purpose: Suggest similar or compatible clothing items.  

- **Virtual Try-On Overlay:**  
  - Algorithm: **Pose Estimation + Image Warping (OpenCV + ML)**  
  - Purpose: Place clothing accurately on user images based on keypoints.  

- **Why these algorithms:**  
  - KNN is simple and effective for similarity matching.  
  - Pose estimation allows realistic virtual try-on.  

---

## Challenges & Difficulties

- **Data Quality:** Images had varying resolutions and backgrounds.  
- **Pose Alignment:** Accurate overlay required careful keypoint detection.  
- **Model Optimization:** Balancing recommendation accuracy with processing speed.  
- **Deployment:** Integrating ML models with Flask while keeping app responsive.  

---

## How to Run

- Clone the repository
- create virtual envirnoment
- Install dependencies as pip install -r req.txt
- Run the application as python app.py

---

##Future Improvements

- Integrate more advanced deep learning models for try-on realism.
- Add user preference tracking to improve recommendations.
- Deploy a cloud-based version for global access.
- Expand dataset to include more clothing types and styles.



git clone https://github.com/siri-orog/Fashion.AI.git
cd Fashion.AI
