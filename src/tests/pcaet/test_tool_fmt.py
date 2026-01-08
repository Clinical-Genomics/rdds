from langchain_ollama import ChatOllama
from rdds.pcaet.ollama import OLLAMA_MODEL, OLLAMA_ENDPOINT
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent

class Response(BaseModel):
    """ If the response is positive or not """
    is_relevant: bool  # Whether the agent responded yes or no

class Colors(BaseModel):
    """ Colors of trees """
    trunk: str  # Color of tree trunk
    leaf: str  # Color of tree leaf

def test():
    model = ChatOllama(model=OLLAMA_MODEL, base_url=f"http://{OLLAMA_ENDPOINT}")
    agent = create_agent(model=model,
                         tools=[],
                         # response_format=ToolStrategy(IsRelevant),
                         response_format=ToolStrategy(Colors),
                         debug=True)
    msg = []
    #msg += [{'role': 'user', 'content': 'Hello are you an agent?'}]
    msg += [{'role': 'user', 'content': 'What are the colors of most tree trunks and leafs?'}]
    r = agent.invoke({'messages': msg})
    r