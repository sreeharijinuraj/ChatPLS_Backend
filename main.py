import os
import PyPDF2
import torch
import psycopg2
import json
import traceback
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import csv


# Initialize Flask app
app = Flask(__name__)

# Load the model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Change to a suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Database configuration
DB_HOST = "aws-0-ap-south-1.pooler.supabase.com"
DB_PORT = "6543"
DB_NAME = "postgres"
DB_USER = "postgres.iuqtgadqkslwohenylab"
DB_PASSWORD = "Sreehari@1234#"  # Replace with your actual password

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    return conn

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text()
        return text_content
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_with_ocr(file):
    try:
        images = convert_from_bytes(file.read())  # Convert PDF pages to images
        text_content = ""
        for image in images:
            text_content += pytesseract.image_to_string(image)
        return text_content
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return ""

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings  # This should be a flat list

def insert_embedding(staff_id, file_name, content, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO embeddings (staff_id, file_name, content, embedding, created_at)
            VALUES (%s, %s, %s, %s::vector, %s)
            """,
            (staff_id, file_name, content, embedding, datetime.now())
        )
        conn.commit()
    except psycopg2.errors.DataError as e:
        print(f"DataError: {e}")
        conn.rollback()  # Rollback the transaction on error
    finally:
        cursor.close()
        conn.close()

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        staff_id = request.form.get('staff_id')

        if not file or not staff_id:
            return jsonify({"error": "File and staff ID are required"}), 400

        # Validate staff_id as numeric (optional)
        if not staff_id.isdigit():
            return jsonify({"error": "Invalid staff ID. Must be numeric."}), 400
        staff_id = int(staff_id)  # Convert to integer if the database expects it

        if file.filename == "":
            return jsonify({"error": "Invalid file name"}), 400

        # Extract text from the uploaded PDF file
        file_content = extract_text_from_pdf(file)

        # If no text was extracted, attempt OCR
        if not file_content.strip():
            file.seek(0)  # Reset file pointer for OCR
            file_content = extract_text_with_ocr(file)

        if not file_content.strip():
            return jsonify({"error": "Could not extract text from the file"}), 400

        # Generate embeddings for the extracted text
        embedding = generate_embeddings(file_content)

        # Insert the embedding into the database
        insert_embedding(staff_id, file.filename, file_content, embedding)

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": file.filename,
            "staff_id": str(staff_id),
            "created_at": str(datetime.now()),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files.get('file')
        staff_id = request.form.get('staff_id')

        if not file or not staff_id:
            return jsonify({"error": "File and staff ID are required"}), 400

        # Validate staff_id as numeric (optional)
        if not staff_id.isdigit():
            return jsonify({"error": "Invalid staff ID. Must be numeric."}), 400
        staff_id = int(staff_id)  # Convert to integer if the database expects it

        # Read the CSV file
        csv_data = file.stream.read().decode('utf-8')  # Decode the file content
        rows = csv_data.splitlines()
        reader = csv.reader(rows)

        # Validate headers
        headers = next(reader, None)
        if 'Content' not in headers or 'Embedding' not in headers:
            return jsonify({"error": "CSV must include 'Content' and 'Embedding' columns"}), 400

        # Get indexes of the relevant columns
        content_idx = headers.index('Content')
        embedding_idx = headers.index('Embedding')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Process rows and insert into database
        for row in reader:
            content = row[content_idx]
            embedding = list(map(float, row[embedding_idx].split(',')))  # Convert embedding to a float vector

            cursor.execute(
                """
                INSERT INTO embeddings (staff_id, file_name, content, embedding, created_at)
                VALUES (%s, %s, %s, %s::vector, %s)
                """,
                (staff_id, file.filename, content, embedding, datetime.now())
            )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "message": "CSV file uploaded and processed successfully",
            "file_name": file.filename,
            "staff_id": str(staff_id),
            "created_at": str(datetime.now()),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500






@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Generate embedding for the query
        query_embedding = generate_embeddings(query)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the embeddings table for cosine similarity
        cursor.execute(
            """
            SELECT content, file_name, staff_id, created_at, embedding <-> %s::vector AS similarity
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY similarity ASC
            LIMIT 5
            """,
            (query_embedding,)
        )

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        # Format the response
        formatted_results = [
            {
                "content": result[0],
                "file_name": result[1],
                "staff_id": result[2],
                "created_at": result[3].strftime('%Y-%m-%d %H:%M:%S'),
                "similarity": result[4],
            }
            for result in results if result[4] is not None
        ]

        return jsonify({"results": formatted_results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)