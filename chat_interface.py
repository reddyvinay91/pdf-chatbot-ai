import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from document_processor import embed_document  # Ensure this line is included

def generate_response(user_query, document_embeddings):
    best_doc_idx = retrieve_answer(user_query, document_embeddings)
    return f"Response from document {best_doc_idx}: (implement your response logic here)"

def retrieve_answer(user_query, document_embeddings):
    query_embedding = embed_document(user_query)  # This should now work
    best_doc_idx = find_best_match(query_embedding, document_embeddings)
    return best_doc_idx

def find_best_match(query_embedding, document_embeddings):
    # Convert to numpy arrays if necessary
    document_embeddings_np = np.array([doc.detach().numpy() for doc in document_embeddings])
    similarities = cosine_similarity(query_embedding.detach().numpy(), document_embeddings_np)
    best_doc_idx = np.argmax(similarities)
    return best_doc_idx
