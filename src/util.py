from typing import Literal
from .state import State
from langgraph.graph import StateGraph, START, END


def next_node(state: State) -> Literal["supervisor", END]:
    if state['continue_loop']:
        return "supervisor"
    else:
        return END