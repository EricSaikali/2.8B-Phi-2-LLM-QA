IMPLEMENTING_RAG = True

import os

if IMPLEMENTING_RAG:
    assert os.path.isdir("model/documents"), "You must have a directory named 'documents' with all the documents used for RAG."

print('Model documents directory exists for the RAG implementation.')
