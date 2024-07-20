from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('tship-f258c-firebase-adminsdk-s7w7w-6f3fe41608.json')  # Update with the path to your service account key
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

@app.route('/similarity', methods=['POST'])
def get_similar_documents():
    data = request.get_json()
    new_description = data.get('description', '')
    
    if not new_description:
        return jsonify({'error': 'No description provided'}), 400
    
    sorted_documents = predict_all_similarities(new_description, descriptions, document_ids, pipeline_nb)
    response_data = [{'document_id': doc_id, 'similarity_score': float(similarity_score)} for doc_id, similarity_score in sorted_documents]
    
    return jsonify({'results': response_data})

if __name__ == '__main__':
    app.run(debug=True)
