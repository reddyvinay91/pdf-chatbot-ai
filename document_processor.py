import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model with a valid model name
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def embed_document(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    embeddings = model(**inputs).pooler_output
    return embeddings
