# logic.py

import os
import logging
import requests
import numpy as np
import cv2
import mediapipe as mp
from werkzeug.utils import secure_filename

IMGBB_API_KEY = '1cf8ce96da2e3643b653d199298425b8'
IMGBB_UPLOAD_URL = 'https://api.imgbb.com/ 1/upload'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def upload_to_imgbb(image_path):
    with open(image_path, 'rb') as image_file:
        payload = {
            'key': IMGBB_API_KEY,
        }
        files = {
            'image': image_file,
        }
        response = requests.post(IMGBB_UPLOAD_URL, data=payload, files=files)
        response_json = response.json()
        if response.status_code == 200 and response_json.get('status') == 200:
            return response_json['data']['url']
        else:
            logging.error(f"ImgBB upload error: {response_json.get('error', 'Unknown error')}")
            return None

def download_file(url, folder='uploads'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(folder, secure_filename(url.split("/")[-1]))
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    else:
        raise Exception("Failed to download file")

def process_image(image_url, known_person_height_cm):
    response = requests.get(image_url)

    if response.status_code == 200:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            def get_landmark_coords(landmark):
                return np.array([landmark.x, landmark.y])

            def calculate_distance(point1, point2):
                return np.linalg.norm(point1 - point2)

            def calculate_measurements(landmarks):
                left_shoulder = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
                right_shoulder = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                left_wrist = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
                right_wrist = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                left_elbow = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
                right_elbow = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
                left_hip = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                right_hip = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                left_ankle = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
                right_ankle = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

                chest_width = calculate_distance(left_shoulder, right_shoulder)
                front_length = calculate_distance(left_shoulder, left_hip)
                sleeve_length = calculate_distance(left_shoulder, left_wrist)
                arm_width = calculate_distance(left_shoulder, left_elbow)

                height = calculate_distance(left_shoulder, left_ankle)

                return chest_width, front_length, sleeve_length, arm_width, height

            chest_width, front_length, sleeve_length, arm_width, height = calculate_measurements(landmarks)

            image_height, image_width, _ = image.shape
            height_pixels = height * image_height
            calibration_factor = known_person_height_cm / height_pixels

            def pixels_to_cm(pixels):
                return pixels * calibration_factor

            chest_width_cm = pixels_to_cm(chest_width * image_width)
            front_length_cm = pixels_to_cm(front_length * image_height)
            sleeve_length_cm = pixels_to_cm(sleeve_length * image_height)
            arm_width_cm = pixels_to_cm(arm_width * image_width)
            score = chest_width_cm + front_length_cm + sleeve_length_cm + arm_width_cm

            return {
                "chest_width": round(chest_width_cm, 2),
                "front_length": round(front_length_cm, 2),
                "sleeve_length": round(sleeve_length_cm, 2),
                "arm_width": round(arm_width_cm, 2),
                "score": round(score, 2)
            }
        else:
            return {"error": "No pose landmarks detected."}
    else:
        return {"error": "Error: Could not download the image. Please check the URL."}

def determine_size(score):
    if 0 <= score <= 200:
        return "S"
    elif 200 < score <= 215:
        return "M"
    elif 215 < score <= 225:
        return "L"
    elif 225 < score <= 240:
        return "XL"
    elif 240 < score <= 255:
        return "XXL"
    elif 255 < score <= 275:
        return "XXXL"
    else:
        return "Sorry, big size not available this time"

# Add other functions like fetch_data_from_firestore and predict_all_similarities here

import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Initialize Firebase Admin SDK
cred = credentials.Certificate("tship-f258c-firebase-adminsdk-s7w7w-6f3fe41608.json")  # Update with the path to your service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Function to fetch data from Firestore
def fetch_data_from_firestore():
    descriptions = []
    document_ids = []

    products_ref = db.collection('products')
    docs = products_ref.get()

    for doc in docs:
        doc_dict = doc.to_dict()
        if 'desc' in doc_dict:
            descriptions.append(doc_dict["desc"])
            document_ids.append(doc.id)
        else:
            descriptions.append('No "desc" field found')
            document_ids.append(doc.id)

    return descriptions, document_ids

# Load data and initialize the model
descriptions, document_ids = fetch_data_from_firestore()

# Initialize your pipeline with existing descriptions
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
pipeline_nb.fit(descriptions, np.random.randint(2, size=len(descriptions)))  # Train with existing descriptions

# Function to predict and sort all documents by similarity
def predict_all_similarities(new_desc, descriptions, document_ids, pipeline):
    new_desc_transformed = pipeline.named_steps['tfidf'].transform([new_desc])
    all_desc_transformed = pipeline.named_steps['tfidf'].transform(descriptions)
    similarities = cosine_similarity(new_desc_transformed, all_desc_transformed)
    sorted_indices = np.argsort(similarities[0])[::-1]
    sorted_docs = [(document_ids[i], similarities[0][i]) for i in sorted_indices]
    return sorted_docs
