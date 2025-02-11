import os
import PyPDF2
import torch
import chromadb
import json
import traceback
import csv
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ChromaDB setup
CHROMA_DB_FOLDER = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
collection = chroma_client.get_or_create_collection(name="product_embeddings")

# Ensure ChromaDB folder exists
if not os.path.exists(CHROMA_DB_FOLDER):
    os.makedirs(CHROMA_DB_FOLDER)

print("ChromaDB collections:", chroma_client.list_collections())


def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_with_ocr(file):
    """Extract text using OCR from an image-based PDF."""
    try:
        images = convert_from_bytes(file.read())
        return "".join(pytesseract.image_to_string(image) for image in images)
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return ""


def generate_embeddings(text):
    """Generate embeddings for the given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload CSV file and store embeddings in ChromaDB."""
    try:
        file = request.files.get('file')

        if not file:
            return jsonify({"error": "CSV file is required"}), 400

        # Read CSV data
        csv_data = file.stream.read().decode('utf-8')
        rows = csv_data.splitlines()
        reader = csv.reader(rows)

        # Validate headers
        headers = next(reader, None)
        if 'Content' not in headers or 'Embedding' not in headers:
            return jsonify({"error": "CSV must include 'Content' and 'Embedding' columns"}), 400

        content_idx = headers.index('Content')
        embedding_idx = headers.index('Embedding')

        # Get existing stored IDs in ChromaDB
        existing_ids = set(collection.get()["ids"])
        print(f"Existing IDs in ChromaDB: {existing_ids}")  # Debugging

        new_entries = []
        new_metadata = []

        for row in reader:
            content = row[content_idx]
            embedding = list(map(float, row[embedding_idx].split(',')))  # Convert to float vector
            unique_id = str(hash(content))  # Unique ID for each row

            # Skip if ID already exists
            if unique_id in existing_ids:
                print(f"Skipping duplicate entry: {content}")
                continue

            new_entries.append(embedding)
            new_metadata.append({"file_name": file.filename, "content": content})

        # Add only new embeddings to ChromaDB
        if new_entries:
            collection.add(
                ids=[str(hash(meta["content"])) for meta in new_metadata],
                embeddings=new_entries,
                metadatas=new_metadata
            )

        return jsonify({
            "message": "CSV file uploaded and stored in ChromaDB successfully",
            "file_name": file.filename,
            'created_at': datetime.now().isoformat(),
            'staff_id': request.form.get('staff_id', 'Unknown')
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/check_file_exists', methods=['GET'])
def check_file_exists():
    """Check if a file already exists in ChromaDB."""
    file_name = request.args.get('file_name')
    if not file_name:
        return jsonify({"error": "Missing file_name parameter"}), 400

    results = collection.get(where={"file_name": file_name})
    return jsonify({"exists": bool(results["ids"])}), (200 if results["ids"] else 404)


@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    """Retrieve uploaded files from ChromaDB."""
    try:
        results = collection.get()
        print("Raw ChromaDB Response:", results)  # Debugging

        uploaded_files = {}
        for i in range(len(results["ids"])):
            file_name = results["metadatas"][i].get("file_name", "Unknown")

            # Avoid duplicates in the response
            if file_name not in uploaded_files:
                uploaded_files[file_name] = {
                    "name": file_name,
                    "dateTime": results["metadatas"][i].get("created_at", "Unknown"),
                    "type": "csv",
                    "staffId": results["metadatas"][i].get("staff_id", "Unknown"),
                }

        print("Filtered API Response:", uploaded_files.values())  # Debugging
        return jsonify(list(uploaded_files.values())), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_file', methods=['DELETE'])
def delete_file():
    data = request.json
    file_name = data.get("file_name")

    if not file_name:
        return jsonify({"error": "File name is required"}), 400

    try:
        # Retrieve IDs associated with the file name
        results = collection.get(where={"file_name": file_name})

        if not results["ids"]:
            return jsonify({"error": "File not found in ChromaDB"}), 404

        # Delete the file from ChromaDB using retrieved IDs
        collection.delete(ids=results["ids"])

        return jsonify({"message": "File deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/search', methods=['POST'])
def search():
    """Search for similar embeddings in ChromaDB."""
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({"error": "Query is required"}), 400

        query_embedding = generate_embeddings(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        formatted_results = [{"content": result["content"], "similarity": similarity}
                             for result, similarity in zip(results["metadatas"][0], results["distances"][0])]

        return jsonify({"results": formatted_results}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
