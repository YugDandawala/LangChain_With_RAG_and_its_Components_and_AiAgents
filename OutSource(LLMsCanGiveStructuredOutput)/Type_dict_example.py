# ✔ A dictionary
# ✔ With predefined keys
# ✔ With enforced data types
# ✔ Defined before the LLM answers
# ✔ Ensured by model-level structured decoding

# if we are only working in python then we should use this but it doesnt have data validation so people not use it more 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="chat-completion"
)
model = ChatHuggingFace(llm=llm)

# Model should be Supportive to this typedict here this model is not so there will be some error  this works directly with models like Chatgpt and etc 
# mostly workds with OpenAi models  even Gemini doesnt support TypeDict

class Review(TypedDict):
     # Data Validation is not possible here for that use Pydantic
        key_themes:Annotated[list[str],'Write down all the key  themes discussed in the review in a list']
        summary:Annotated[str,'A brief summary of the review']
        sentiment:Annotated[Literal['pos','neg'],'Return sentiment of review either negative or positive']
        # Because of this literal it will answer in pos or negative only like (pos or neg) otherwise if it is as others it will give a proper message for this
        pros:Annotated[Optional[list[str]],'write down all pros inside the list']
        cons:Annotated[Optional[list[str]],'write down all cons inside the list']
        name:Annotated[Optional[list[str]],'write name of reviewer']
        
structured_model=model.with_structured_output(Review)

result=structured_model.invoke(""" The Hardware is great very good and performance level processor with high end GPU and very good OS system of the whole device.
                                                                       Pros:Very good .    Cons:not good""")

print(result)
print(result['summary'])
print(result['sentiment'])

# For typedict in the keys name will be there but in output it will be not as in review name is not there but if it is it will show as it is optional
print(result.keys())