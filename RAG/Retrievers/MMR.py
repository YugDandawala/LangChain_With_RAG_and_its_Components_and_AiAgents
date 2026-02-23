# Because of MMR we can pick results that are not only relevant to query but also different from each other

# Here we have used retriever because the vector store uses semantic search by just one algo of finding euclidien distance similarity
# But if we user some more advanced retrievers with vector store the retriever do semantci search with more types of internal algos which makes it more better

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
# Create embeddings using HuggingFace Inference API
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="example_collection",
)
retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance 
    # it lambda_mult  is 1 it will behave as pure semantic search of vector store and if 0 it will be pure diverse so mainly the value should be in range of 0-1
)
query = "What is Chroma in langchain"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)