# This is for downlaoding the emebdding models only not the WHole LLM model

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text="Delhi is capital of india"
# vector=embeddings.embed_query(text)

documents=["Delhi is capital","surat is city in gujarat","vadodara is also"]
vector=embeddings.embed_documents(documents)

print(str(vector))
