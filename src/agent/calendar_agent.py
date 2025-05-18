from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import json
from datetime import datetime, timedelta

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']

class CalendarEvent(BaseModel):
    """Represents a calendar event."""
    summary: str = Field(..., description="Title of the event")
    start_time: str = Field(..., description="Start time of the event in ISO format")
    end_time: str = Field(..., description="End time of the event in ISO format")
    timezone: str = Field(default="UTC", description="Timezone for the event")
    description: Optional[str] = Field(default=None, description="Optional description")
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee emails")


class CalendarAgent:
    """Agent for handling calendar operations like scheduling and checking availability.
    
    This agent uses LangChain's Google Calendar integration to interact with Google Calendar.
    It provides a simplified interface for common calendar operations.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the CalendarAgent.
        
        Args:
            llm: Optional ChatOpenAI instance. If not provided, a default one will be created.
        """
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        self.credentials = None
        self.service = None
        
        # Initialize Google Calendar service
        self._initialize_calendar_service()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create the agent
        self.agent = self._create_agent()
        
    def _initialize_calendar_service(self):
        """Initialize the Google Calendar service."""
        creds = None
        token_path = 'token.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=49732)
            
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.credentials = creds
        self.service = build('calendar', 'v3', credentials=creds)
        
    def _create_tools(self) -> list:
        """Create the tools for the agent."""
        return [
            Tool(
                name="create_calendar_event",
                func=self._create_event,
                description="""Useful for creating a new calendar event. 
                Input should be a JSON string with the following keys:
                'summary': str (required) - The event title
                'start_time': str (required) - Start time in ISO format
                'end_time': str (required) - End time in ISO format
                'description': str (optional) - Event description
                'location': str (optional) - Event location
                'attendees': List[str] (optional) - List of attendee emails
                """
            ),
            Tool(
                name="list_calendar_events",
                func=self._list_events,
                description="""Useful for listing calendar events.
                Input should be a JSON string with the following optional keys:
                'max_results': int - Maximum number of events to return (default: 10)
                'time_min': str - Start time in ISO format (default: now)
                'time_max': str - End time in ISO format
                """
            )
        ]
        
    def _create_event(self, event_data: str) -> str:
        """Create a new calendar event."""
        try:
            data = json.loads(event_data)
            event = {
                'summary': data['summary'],
                'start': {'dateTime': data['start_time'], 'timeZone': 'UTC'},
                'end': {'dateTime': data['end_time'], 'timeZone': 'UTC'},
            }
            
            if 'description' in data:
                event['description'] = data['description']
            if 'location' in data:
                event['location'] = data['location']
            if 'attendees' in data:
                event['attendees'] = [{'email': email} for email in data['attendees']]
            
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            return f"Event created: {event.get('htmlLink')}"
            
        except Exception as e:
            return f"Error creating event: {str(e)}"
            
    def _list_events(self, params: str) -> str:
        """List calendar events."""
        try:
            params = json.loads(params) if params else {}
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=params.get('time_min', now),
                timeMax=params.get('time_max'),
                maxResults=params.get('max_results', 10),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            if not events:
                return 'No upcoming events found.'
                
            return '\n'.join(
                f"{e['start'].get('dateTime', e['start'].get('date'))} - {e['summary']}"
                for e in events
            )
            
        except Exception as e:
            return f"Error listing events: {str(e)}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the calendar agent with tools and prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful calendar assistant. Your job is to help users schedule 
            meetings and check availability. Always be polite and professional. If you need more 
            information to complete a request, ask follow-up questions.
            
            You have access to the following tools:
            - create_calendar_event: Create a new calendar event
            - list_calendar_events: List upcoming calendar events
            
            When creating events, make sure to include all necessary details like title, 
            start/end times, and any attendees. For checking availability, you can list events 
            within a specific time range."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the calendar-related request with proper chat format for LangGraph Studio.
        
        This method handles the conversation flow, processes user input, and interacts with
        Google Calendar through LangChain tools. It manages authentication, error handling,
        and maintains conversation context.
        
        Args:
            state: Dictionary containing the current state with messages and other context.
                Expected keys:
                - messages: List of message dicts with 'role' and 'content' keys
                - intent: Optional intent of the conversation
                - booking_details: Optional dict with booking-related details
                
        Returns:
            Dict containing the updated state with the assistant's response and metadata.
        """
        # Initialize messages if not present
        messages = state.get("messages", [])
        if not messages:
            welcome_msg = "I'm here to help with calendar scheduling. How can I assist you today?"
            return {
                "messages": [{"role": "assistant", "content": welcome_msg}],
                "response": welcome_msg,
                "user_query": "",
                "intent": state.get("intent", ""),
                "booking_details": state.get("booking_details", {})
            }
        
        # Get the last user message
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return state
            
        user_input = last_message.get("content", "").strip()
        if not user_input:
            return state
        
        # Format chat history for the agent
        formatted_history = []
        for msg in messages[:-1]:  # Exclude the last message as it's the current input
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))
        
        try:
            # Check if we need to handle authentication flow
            if not self.is_authenticated():
                auth_msg = (
                    "I need to authenticate with Google Calendar first. "
                    "Please provide the necessary permissions to access your calendar. "
                    "You'll be redirected to Google's authentication page."
                )
                return {
                    "messages": [*messages, {"role": "assistant", "content": auth_msg}],
                    "response": auth_msg,
                    "user_query": user_input,
                    "intent": state.get("intent", ""),
                    "booking_details": state.get("booking_details", {}),
                    "needs_authentication": True
                }
            
            # Process the user input with the agent
            try:
                result = await self.agent.ainvoke({
                    "input": user_input,
                    "chat_history": formatted_history,
                })
                response = result.get("output", "I'm not sure how to respond to that.")
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(auth_term in error_msg for auth_term in ["invalid_grant", "credentials", "authentication"]):
                    response = (
                        "I need to re-authenticate with Google Calendar. "
                        "Please provide the necessary permissions to continue."
                    )
                    return {
                        "messages": [*messages, {"role": "assistant", "content": response}],
                        "response": response,
                        "user_query": user_input,
                        "intent": state.get("intent", ""),
                        "booking_details": state.get("booking_details", {}),
                        "needs_authentication": True
                    }
                else:
                    raise e
            
            # Update state with the response
            return {
                "messages": [*messages, {"role": "assistant", "content": response}],
                "response": response,
                "user_query": user_input,
                "intent": state.get("intent", ""),
                "booking_details": state.get("booking_details", {})
            }
            
        except Exception as e:
            import traceback
            error_msg = (
                "I'm sorry, I encountered an error while processing your request. "
                f"Please try again or rephrase your request. Error: {str(e)}"
            )
            print(f"Error in calendar_agent.process(): {str(e)}\n{traceback.format_exc()}")
            
            return {
                "messages": [*messages, {"role": "assistant", "content": error_msg}],
                "response": error_msg,
                "user_query": user_input,
                "intent": state.get("intent", ""),
                "booking_details": state.get("booking_details", {}),
                "error": str(e)
            }
    
    def is_authenticated(self) -> bool:
        """Check if the agent is authenticated with Google Calendar."""
        return self.credentials is not None and self.credentials.valid


# Singleton instance
calendar_agent = CalendarAgent()
