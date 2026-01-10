from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import tool

from src.nutrition.nutrition_tools import NutritionTools

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]




def build_nutrition_agent_system():
     # Pass the local path to the model
    chat_model = init_chat_model(f"{MODEL_NAME}", temperature=0.2)


    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, NutritionTools)
    print("LangGraph agent created!")
    print(react_agent)

    MAX_TURNS = 20 
    return react_agent
