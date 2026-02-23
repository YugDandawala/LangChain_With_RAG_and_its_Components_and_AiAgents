import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceInferenceEmbeddings

load_dotenv()

# Make sure your HF token is set
print("HF_TOKEN =", os.getenv("HF_TOKEN"))

# Create embeddings using HuggingFace Inference API
embeddings = HuggingFaceInferenceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = ["Delhi is capital", "Surat is city in Gujarat", "Vadodara is also"]
vectors = embeddings.embed_documents(documents)

print(vectors)
