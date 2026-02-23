# Example of some user asking some question to model

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

# # for single query
# result=model.embed_query("What is capital of india")

documents=["Virat kohli is an indian cricketer know for aggression",
           "Ms dhoni is an indian cricketer know for his coolness",
           "Rohit is an indian cricketer know for his funnyness",
           "Hardik is an indian cricketer know for his Hitting",
           ]

query="tell me about virat kohli"

# embedded it and store edit vectors in database
doc_embeddings=model.embed_documents(documents)
doc_query=model.embed_query(query)

scores=cosine_similarity([doc_query],doc_embeddings)

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
# Similarity Score
print("similar score is :",score)