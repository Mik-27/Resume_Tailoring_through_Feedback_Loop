from typing import Literal
from .state import State
from langgraph.graph import StateGraph, START, END


def next_node(state: State) -> Literal["Supervisor", END]:
    if state['continue_loop']:
        return "Supervisor"
    else:
        return END