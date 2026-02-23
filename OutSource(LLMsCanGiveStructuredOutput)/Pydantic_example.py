# Pydantic is a Python library for data validation and settings management using Python type hints. It lets you define data models (like schemas) and ensures that any data you create matches the expected types and structure.

# if we are only working in python then we should use this more
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Optional,Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = HuggingFaceEndpoint(
    # Function calling is manily recommended to use for OpenAI models  and this helps in making { Agents }
    repo_id="openai/gpt-oss-120b",
    task="chat-completion"
)
model = ChatHuggingFace(llm=llm)

# Here there will be a error like pydantic doesnt support function calling because huggingface  models doesnt support function calling its mainly avaible in OpenAI models and etc big models

class Review(BaseModel):
     
        key_themes:list[str]=Field(description="Write down all the key  themes discussed in the review in a list")
        summary:str=Field(description="A brief summary of the review")
        sentiment:Literal["pos","neg"]=Field(description="Return sentiment of review either negative or positive")
        # Because of this literal it will answer in pos or negative only like (pos or neg) otherwise if it is as others it will give a proper message for this
        pros:Optional[list[str]]=Field(default=None,description="write down all pros inside the list")
        cons:Optional[list[str]]=Field(default=None,description="write down all cons inside the list")
        name:Optional[str]=Field(default=None,description="write name of reviewer")
        
structured_model=model.with_structured_output(Review)

result=structured_model.invoke(""" This is a great phone with excellent performance and a fantastic display, though some minor issues with the camera focus and pre-installed apps prevent it from being perfect. Overall, it's a highly recommended device for anyone seeking a reliable and refined user experience. 
Pros  -->
Excellent Performance: The phone's processor ensures smooth, fast, and responsive performance for all your daily tasks.
Superb Display: The display is bright, sharp, and vibrant, making it great for watching videos and general use.
Reliable Camera: The camera system is consistently strong in most situations, delivering high-quality photos.
Strong Battery Life: The battery provides excellent endurance, easily lasting through a full day of use. 
Cons  -->
Camera Focus Issues: The camera can sometimes have slight focus problems, which may require a bit of patience to get a perfect shot.
Minor Pre-installed Apps: There are a few unnecessary pre-installed apps (bloatware) that you'll need to remove to clean up the experience. """)

result_dict=dict(result)
result_json=result.model_dump_json()

print(result)
print(result.name)
print(result.sentiment)

print(result_dict['age'])

print(result_json)
