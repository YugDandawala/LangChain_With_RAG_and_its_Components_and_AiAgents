# Here we have used retriever because the vector store uses semantic search by just one algo of finding euclidien distance similarity
# But if we user some more advanced retrievers with vector store the retriever do semantci search with more types of internal algos which makes it more better

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
# Create embeddings using HuggingFace Inference API
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="example_collection",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma in langchain"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)