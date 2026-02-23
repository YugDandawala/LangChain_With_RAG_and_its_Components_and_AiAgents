from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import requests

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=api_key&query={city}"
    response = requests.get(url)
    return response.json


model = OllamaLLM(model="llama3.2", temperature=0.1)

# hwchase17/react (hub not loading need langsmith api key) so use prompt from the hub
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
**Action Input must be a valid JSON object**
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)

agent = create_react_agent(
    llm=model, tools=[search_tool, get_weather_data], prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)

response = agent_executor.invoke(
    {
        "input": "What is the current weather conditin of surat need All the Data.If there is pollution what is the source of it research and give the solution of it"
    }
)
print(response)
