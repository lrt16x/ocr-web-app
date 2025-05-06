
import os, io, tempfile, json, base64, uuid
from flask import Flask, request, render_template, send_file, jsonify
from pdf2image import convert_from_bytes
from google.cloud import vision

app = Flask(__name__)

# expects GOOGLE_APPLICATION_CREDENTIALS env var path to JSON key
client = vision.ImageAnnotatorClient()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ocr", methods=["POST"])
def ocr_pdf():
    if 'file' not in request.files:
        return jsonify({'error':'No file'}), 400
    pdf_file = request.files['file'].read()
    pages = convert_from_bytes(pdf_file, dpi=300)
    combined_text = []
    for idx, img in enumerate(pages, start=1):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        content = buf.getvalue()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        text = response.full_text_annotation.text if response.full_text_annotation.text else ''
        combined_text.append(f"--- PAGE {idx} ---\n{text.strip()}\n")
    result = "\n".join(combined_text)
    # return as text file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    tmp.write(result.encode('utf-8'))
    tmp.flush()
    return send_file(tmp.name, as_attachment=True, download_name='ocr_output.txt', mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)

