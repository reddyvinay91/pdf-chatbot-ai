import unittest
from document_processor import extract_text_from_pdf, embed_document
from chat_interface import retrieve_answer, generate_response

class TestPdfChatApp(unittest.TestCase):

    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf("sample.pdf")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_embed_document(self):
        embeddings = embed_document("Sample text for testing.")
        self.assertEqual(len(embeddings.shape), 2)

    def test_generate_response(self):
        # Mocking a simple retrieval
        response = generate_response("What is AI?", ["Artificial Intelligence is..."], embed_document("Artificial Intelligence is..."))
        self.assertIsInstance(response, str)

if __name__ == "__main__":
    unittest.main()
