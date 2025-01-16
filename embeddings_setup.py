# embeddings_setup.py

from langchain_huggingface import HuggingFaceEmbeddings

LOCAL_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDINGS_MODEL)
