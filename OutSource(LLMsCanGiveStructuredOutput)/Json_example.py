# it can be used when multiple langauges are there in our project so it so main

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import json

load_dotenv()

# Load the schema from the JSON file
with open('Json_Review.json', 'r') as f:
    Json_review = json.load(f)
    
# Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

        
structured_model=model.with_structured_output(Json_review)

result=structured_model.invoke(""" The Hardware is great very good and performance level processor with high end GPU and very good OS system of the whole device.
                                                                       Pros:Very good .    Cons:not good""")

print(result)
