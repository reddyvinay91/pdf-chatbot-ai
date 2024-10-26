import streamlit as st
from document_processor import extract_text_from_pdf, embed_document
from chat_interface import generate_response

st.title("AI Chat with PDF Knowledge Base")
st.write("Upload PDF(s)")

# Upload PDF file
uploaded_file = st.file_uploader("Drag and drop files here", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:")
    st.write(text)
    
    # Generate embeddings
    embeddings = embed_document(text)
    st.write("Embeddings generated.")

    # User question input
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        response = generate_response(user_query, embeddings)
        st.write("Response:")
        st.write(response)
