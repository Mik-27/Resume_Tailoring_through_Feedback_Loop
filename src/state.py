from typing import TypedDict, Annotated, List
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph.message import add_messages


# Class storing the state structure for thr graph
class State(TypedDict):
    resume: str
    job_description: str
    messages: Annotated[List[HumanMessage], add_messages]
    iteration: int
    feedback: str
    agent_outputs: Annotated[List[AIMessage], add_messages]
    relevancy: float
    continue_loop: bool