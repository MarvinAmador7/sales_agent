"""Sales Conversation Supervisor Graph.

This module implements a supervisor agent that handles sales conversations,
including product questions and demo bookings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict, Annotated, Optional, Union, cast
from typing_extensions import TypedDict as TypedDictExt

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

# Import the calendar agent
from src.agent.calendar_agent import calendar_agent

# Initialize the LLM with streaming enabled for chat
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    streaming=True  # Enable streaming for chat
)


class AgentState(TypedDict):
    """State for the sales conversation agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    intent: Literal["product_question", "book_demo", "escalate", "end"]
    response: str
    booking_details: Optional[Dict[str, str]] = None


async def supervisor_node(state: AgentState) -> AgentState:
    # Get the last user message
    last_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None
    )
    
    if last_message:
        state["user_query"] = last_message.content
    
    # Classify the intent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sales conversation classifier. 
        Classify the user query into one of the following intents:
        - product_question: Questions about product features or capabilities
        - book_demo: Request to schedule a demo
        - escalate: Requests to speak with a human
        - end: User wants to end the conversation"
        
        User query: {user_query}
        
        Respond with ONLY the intent name from the list above."""),
        ("human", "{user_query}")
    ])
    
    # Get the intent classification
    intent = await llm.ainvoke(prompt.format_messages(user_query=state["user_query"]))
    state["intent"] = intent.content.strip()
    
    # Add the assistant's classification as a system message
    state["messages"].append({
        "type": "system",
        "content": f"Classified intent: {state['intent']}"
    })
    
    return state
    

async def knowledge_base_node(state: AgentState) -> AgentState:
    """Node for handling product-related questions."""
    # In a real implementation, this would query a knowledge base
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful sales assistant. Answer the user's question about the product."),
        *state["messages"][-5:],  # Use last 5 messages for context
    ])
    
    state["response"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    return state



async def calendar_node(state: AgentState) -> AgentState:
    """Node for handling calendar operations using the CalendarAgent."""
    try:
        # Ensure calendar agent is properly initialized
        if not hasattr(calendar_agent, 'ensure_authenticated'):
            error_msg = "Calendar service is not available. Please try again later."
            state["messages"].append(AIMessage(content=error_msg))
            state["response"] = error_msg
            return state

        # Get the last user message
        last_message = next(
            (msg for msg in reversed(state["messages"]) 
             if isinstance(msg, HumanMessage)),
            None
        )

        if not last_message:
            error_msg = "I didn't catch that. Could you please repeat your request?"
            state["messages"].append(AIMessage(content=error_msg))
            state["response"] = error_msg
            return state

        # Process the calendar request
        response = await calendar_agent.invoke({"input": last_message.get("content", "")})
        response_content = response.get("output", "I've processed your calendar request.")

        # Update state with response
        state["messages"].append(AIMessage(content=response_content))
        state["response"] = response_content

    except Exception as e:
        error_msg = "I'm having trouble accessing the calendar. Please try again later."
        print(f"Error in calendar_node: {e}", exc_info=True)
        state["messages"].append(AIMessage(content=error_msg))
        state["response"] = error_msg

    return state


async def whatsapp_node(state: AgentState) -> AgentState:
    """Node for sending responses via WhatsApp."""
    # In a real implementation, this would send the response via WhatsApp
    print(f"Sending WhatsApp message: {state['response']}")
    # Add the response to messages for chat history
    state["messages"].append(AIMessage(content=state["response"]))
    return state


async def escalate_node(state: AgentState) -> AgentState:
    """Node for escalating to a human agent."""
    state["messages"].append(SystemMessage(content="Escalating to human agent..."))
    state["response"] = "I'll connect you with a human agent who can better assist you."
    state["intent"] = "end"
    return state


# Create the workflow
workflow = StateGraph(AgentState)

# Add the entry point for chat messages
workflow.add_edge(START, "supervisor")

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("knowledge_base", knowledge_base_node)
workflow.add_node("calendar", calendar_node)
workflow.add_node("whatsapp", whatsapp_node)
workflow.add_node("escalate", escalate_node)

# Define the routing logic
def route_to_node(state: AgentState) -> str:
    """Route to the appropriate node based on intent."""
    if state["intent"] == "product_question":
        return "knowledge_base"
    elif state["intent"] == "book_demo":
        return "calendar"
    elif state["intent"] == "escalate":
        return "escalate"
    return "whatsapp"  # Default node for responses

# Add edges
workflow.add_conditional_edges(
    "supervisor",
    route_to_node,
    {
        "knowledge_base": "knowledge_base",
        "calendar": "calendar",
        "escalate": "escalate",
        "whatsapp": "whatsapp"
    }
)

# Connect all nodes back to whatsapp for response
for node in ["knowledge_base", "calendar", "escalate"]:
    workflow.add_edge(node, "whatsapp")

# Connect whatsapp to END to complete the cycle
workflow.add_edge("whatsapp", END)

# Set entry point
workflow.set_entry_point("supervisor")

# Compile the graph
graph = workflow.compile()
