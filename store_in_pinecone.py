# store_in_pinecone.py

import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore as LCPinecone
from langchain.schema import Document
from embeddings_setup import embeddings
import PyPDF2
import pinecone

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
)

index_name = "my-hybrid-index"  
all_indexes = pc.list_indexes().names()

if index_name not in all_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,         
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.environ.get("PINECONE_ENV", "us-east-1")
        )
    )


info = pc.describe_index(index_name)
print("Index info:", info)

host = info["host"]
if not host:
    raise ValueError("Index host is missing. Check your index status or region in Pinecone.")

index = Index(
    api_key=os.environ["PINECONE_API_KEY"],
    host=host,
    name=index_name
)

# Clear old data
try:
    index_stats = index.describe_index_stats()
    existing_namespaces = index_stats.get('namespaces', {})
    
    if existing_namespaces:
        for ns in existing_namespaces:
            index.delete(delete_all=True, namespace=ns)
            print(f"Deleted all vectors in namespace '{ns}'.")
    else:
        print("No namespaces found to delete.")
except pinecone.core.client.exceptions.NotFoundException:
    print("Namespace not found. No existing documents to delete.")
except Exception as e:
    print(f"Error deleting documents from Pinecone: {e}")

# Upload new documents
docs_path = "docs"
if not os.path.exists(docs_path):
    raise ValueError(f"Directory {docs_path} does not exist. Please add your documents.")

import PyPDF2  # Ensure PyPDF2 is installed: pip install PyPDF2

docs_texts = []
supported_extensions = ('.txt', '.md', '.pdf')

for file_name in os.listdir(docs_path):
    file_path = os.path.join(docs_path, file_name)
    
    if not os.path.isfile(file_path):
        print(f"Skipping '{file_name}': Not a file.")
        continue

    if file_name.lower().endswith(('.txt', '.md')):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
            docs_texts.append(text)
            print(f"Successfully read '{file_name}'.")
        except Exception as e:
            print(f"Error reading '{file_name}': {e}")
    
    elif file_name.lower().endswith('.pdf'):
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text:
                docs_texts.append(text)
                print(f"Successfully extracted text from '{file_name}'.")
            else:
                print(f"No text found in '{file_name}'.")
        except Exception as e:
            print(f"Error processing PDF '{file_name}': {e}")
    
    else:
        print(f"Unsupported file type for '{file_name}'. Skipping.")

documents = [Document(page_content=txt) for txt in docs_texts]
vectorstore = LCPinecone.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name="my-hybrid-index"
)
print("Documents uploaded to Pinecone!")