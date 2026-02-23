from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

# # for single query
# result=model.embed_query("What is capital of india")

documents=["Delhi is capital","surat is city in gujarat","vadodara is also"]
result=model.embed_documents(documents)

print(str(result))
# This will give the numerical vector for this input query 