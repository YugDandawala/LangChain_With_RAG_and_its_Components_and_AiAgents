from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

user = "postgres"
password = quote_plus("Admin@123")
host = "localhost"
database = "langchain"

collection = "embedded_vectors"
connectDB = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.8
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Generate a proper detailed report about nature")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=15
)
text=splitter.split_text(result.content)

# Create embeddings using HuggingFace Inference API
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embeddings.embed_documents(text)

docs = [Document(page_content=t, metadata={"embedding": v}) for t, v in zip(text, vectors)]

db = PGVector(
    embeddings = embeddings,
    collection_name = collection,
    connection = connectDB
)

# It will create 2 tables in pgadmin  ( langchain_pg_collection  and  langchain_pg_embedding )
# In langchain_pg_collection a table with its uuid is stored in this 
# In langchain_pg_embedding all the vectors with their Id,text and metadeta  is stored in it 
    
db.add_documents(docs)

query = "Tell me the beauty of mountains"
results = db.similarity_search(query, k=100) 
for doc in results:
    print(doc.page_content)

delete = db.delete(ids=['0de0f3db-40de-49db-8cc6-b1740833c565','ab5e0162-8447-4eb5-8906-42725f0b58be'])
print(delete)