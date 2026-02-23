from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-4",temperature=1,max_completion_tokens=50)

# if temperature is 0 then alltime when we run the code it will give the same output for the same input 
# temperature is a paramater is of range 0-2 and are based on random and creative
# for logical and normal QA  rangeo 0-0.8 but storytelling etc 0.8-1 and more for creative more than it 
# how many tokens you want or roughly how many words you need as the Api is paid 

result=model.invoke("what is the capital of india")

print(result.content)