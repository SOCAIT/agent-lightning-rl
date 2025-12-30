from langgraph.graph import StateGraph, START, END
from loa
from dotenv import load_dotenv
load_dotenv()

import os

from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import tool

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]



def build_nutrition_agent_system():
    model
    
