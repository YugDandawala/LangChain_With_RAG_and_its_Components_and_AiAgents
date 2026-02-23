# It find the similiraity between the text sentences and check the similarity and through which it divides the chunks
# This method is removed by langchain in the latest version so you have do it manually by RecursiveCharacterTextSplitter,HuggingFaceEmbeddings,cosine_similarity as done in  <--- example.py in EmbeddedModels --->
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

text="""
    Space exploration has advanced rapidly over the past few decades, with new missions aiming to study distant planets, asteroids, and the search for extraterrestrial life. Scientists are particularly interested in understanding how galaxies form and what conditions are necessary for life to exist beyond Earth. At the same time, many people are turning to gardening as a way to relax, grow their own food, and connect with nature. Home gardens filled with herbs, vegetables, and flowers provide both therapeutic benefits and a sustainable source of fresh produce.
    
    Modern digital technology has transformed how people work, communicate, and access information, with smartphones and cloud platforms becoming essential tools in daily life. Advances in artificial intelligence and automation are reshaping industries and creating new opportunities for innovation. Meanwhile, wildlife conservation efforts focus on protecting endangered species and restoring natural habitats threatened by climate change, deforestation, and illegal hunting. Conservationists use scientific research and community programs to preserve biodiversity and maintain ecological balance.
"""
splitter=SemanticChunker(
    model,breakpoint_threshold_type="Gradient",
    breakpoint_threshold_amount=1
)

# result=splitter.split_documents(docs)
result=splitter.create_document([text])
print(len(result))
print(result)