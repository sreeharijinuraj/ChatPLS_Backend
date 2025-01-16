import os
import PyPDF2
import torch
import psycopg2
import traceback
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Change to a suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Database configuration
DB_HOST = os.getenv("DB_HOST", "aws-0-ap-south-1.pooler.supabase.com")
DB_PORT = os.getenv("DB_PORT", "6543")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.iuqtgadqkslwohenylab")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Sreehari@1234#")  # Replace with your actual password



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
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # Attempt to decrypt if encrypted
            except Exception as decrypt_error:
                print(f"Failed to decrypt PDF: {decrypt_error}")
                return ""

        text_content = ""
        for page_number, page in enumerate(reader.pages):
            try:
                text_content += page.extract_text() or ""  # Attempt text extraction
            except Exception as page_error:
                print(f"Error on page {page_number}: {page_error}")
        return text_content

    except Exception as e:
        print(f"Error reading PDF: {e}")
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

def generate_embeddings_parallel(text_chunks):
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(generate_embeddings, text_chunks))
    return torch.mean(torch.tensor(embeddings), dim=0).tolist()

def insert_image_embedding(doc_id, page_number, image_number, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO embeddings_images (doc_id, page_number, image_number, embedding, created_at)
            VALUES (%s, %s, %s, %s::vector, %s)
            """,
            (doc_id, page_number, image_number, embedding, datetime.now())
        )
        conn.commit()
    except psycopg2.errors.DataError as e:
        print(f"DataError: {e}")
        conn.rollback()
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

        if not staff_id.isdigit():
            return jsonify({"error": "Invalid staff ID. Must be numeric."}), 400
        staff_id = int(staff_id)

        if file.filename == "":
            return jsonify({"error": "Invalid file name"}), 400

        # Extract text from the uploaded PDF file
        file_content = extract_text_from_pdf(file)

        # If no text was extracted, attempt OCR
        if not file_content.strip():
            file.seek(0)
            file_content = extract_text_with_ocr(file)

        if not file_content.strip():
            return jsonify({"error": "Could not extract text from the file"}), 400

        # Split text into smaller chunks for parallel embedding generation
        text_chunks = [file_content[i:i + 500] for i in range(0, len(file_content), 500)]
        text_embedding = generate_embeddings_parallel(text_chunks)

        # Insert the text embedding into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO embeddings (staff_id, file_name, content, created_at)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (staff_id, file.filename, file_content, datetime.now())
        )
        doc_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()


        # Extract image embeddings
        file.seek(0)  # Reset file pointer
        images = convert_from_bytes(file.read())
        for page_number, image in enumerate(images):
            image_number = 0  # Reset image number for each page

            # Option 1: Use entire image for embedding
            image_embedding = generate_embeddings(pytesseract.image_to_string(image))
            insert_image_embedding(doc_id, page_number, image_number, image_embedding)

            # Increment the image number for consistency
            image_number += 1

        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_name": file.filename,
            "staff_id": str(staff_id),
            "created_at": str(datetime.now()),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Search Endpoint
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
        cursor.execute(
            """
            SELECT content, file_name, staff_id, created_at, embedding <-> %s::vector AS similarity
            FROM embeddings
            ORDER BY similarity
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
            for result in results
        ]

        return jsonify({"results": formatted_results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)