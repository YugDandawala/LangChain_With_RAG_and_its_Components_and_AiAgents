# When you will run it the model will be installed in our System locally and will reply with the output
# the model will be stored in D drive but 2 line of code

import os
os.environ['HF_Home']="D:/Huggingface_cache"

from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline.from_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",task="chat-completion",
    pipeline_kwargs=dict(temperature=0.5,max_new_tokens=100)
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("Write a poem for football in 2 lines")

print(result.content)

