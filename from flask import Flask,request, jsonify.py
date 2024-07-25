# main.py

from flask import Flask, request, jsonify
import logging
from gradio_client import Client, file
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from logic import upload_to_imgbb, download_file, process_image, determine_size, fetch_data_from_firestore, predict_all_similarities

app = Flask(__name__)

# Load data and initialize the model
descriptions, document_ids = fetch_data_from_firestore()

# Initialize your pipeline with existing descriptions
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
pipeline_nb.fit(descriptions, np.random.randint(2, size=len(descriptions)))  # Train with existing descriptions

@app.route('/similarity', methods=['POST'])
def get_similar_documents():
    data = request.get_json()
    new_description = data.get('description', '')
    
    if not new_description:
        return jsonify({'error': 'No description provided'}), 400
    
    sorted_documents = predict_all_similarities(new_description, descriptions, document_ids, pipeline_nb)
    response_data = [{'document_id': doc_id, 'similarity_score': float(similarity_score)} for doc_id, similarity_score in sorted_documents]
    
    return jsonify({'results': response_data})

@app.route('/all', methods=['GET'])
def get_similar_documents_get():
    new_description = request.args.get('description', 'all')
    
        
    sorted_documents = predict_all_similarities(new_description, descriptions, document_ids, pipeline_nb)
    response_data = [{'document_id': doc_id, 'similarity_score': float(similarity_score)} for doc_id, similarity_score in sorted_documents]
    
    return jsonify({'results': response_data})

client = Client("yisol/IDM-VTON")

@app.route('/process', methods=['POST'])
def process():
    # Get the JSON data from the request
    data = request.get_json()
    background_url = data.get('background')
    garm_img_url = data.get('garm_img')

    if background_url and garm_img_url:
        try:
            # Download the files from the URLs
            background_file_path = download_file(background_url)
            garm_img_file_path = download_file(garm_img_url)

            # Call the Gradio client prediction
            result = client.predict(
                dict={
                    "background": file(background_file_path),
                    "layers": [],
                    "composite": None
                },
                garm_img=file(garm_img_file_path),
                garment_des="Hello!!",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            
            # Return the result
            output_url = upload_to_imgbb(result[0])
            return jsonify({"output_file": output_url})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "URLs not found in request"}), 400

@app.route('/measurements', methods=['POST'])
def measurements():
    data = request.json
    image_url = data.get('image')
    known_person_height_cm = data.get('height')

    if not image_url or not known_person_height_cm:
        return jsonify({"error": "Missing image_url or known_person_height_cm"}), 400

    result = process_image(image_url, known_person_height_cm)
    
    if "error" not in result:
        result["size"] = determine_size(result["score"])
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
