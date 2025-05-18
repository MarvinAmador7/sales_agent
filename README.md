# Sales Agent with Calendar Integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A sophisticated sales assistant agent built with LangGraph that handles customer inquiries, schedules demos, and manages calendar integration. The agent can understand natural language requests, check availability, and schedule meetings using Google Calendar.

## ✨ Features

- **Natural Language Understanding**: Processes customer inquiries using OpenAI's GPT models
- **Calendar Integration**: Seamlessly schedules and manages appointments
- **Conversation Flow**: Maintains context across multi-turn conversations
- **Modular Design**: Easy to extend with new capabilities
- **Visual Debugging**: Built-in support for LangGraph Studio

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) for dependency management
- OpenAI API key
- Google Cloud credentials (for Calendar integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sales-agent.git
   cd sales-agent
   ```

2. **Install dependencies**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install project dependencies
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file with your API keys and configuration:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key
   
   # Optional but recommended
   LANGSMITH_API_KEY=your_langsmith_key  # For tracing and monitoring
   GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json  # For Calendar integration
   ```

### Running the Agent

1. **Start the development server**
   ```bash
   poetry run langgraph dev
   ```

2. **Access LangGraph Studio**
   Open your browser to `http://localhost:2024` to interact with the agent through the LangGraph Studio interface.

## 🛠️ Project Structure

```
sales-agent/
├── .github/               # GitHub Actions workflows
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── graph.py         # Main agent graph definition
│       └── calendar_agent.py # Calendar integration logic
├── tests/                  # Test files
├── .env.example           # Example environment variables
├── poetry.lock            # Locked dependencies
└── pyproject.toml         # Project configuration
```

## 🤖 Using the Agent

### Example Conversations

**Scheduling a Demo**
```
User: Can we schedule a demo for next Tuesday at 2pm?
Agent: I've scheduled your demo for Tuesday, [date] at 2:00 PM. You'll receive a calendar invite shortly.
```

**Checking Availability**
```
User: What's your availability tomorrow?
Agent: I'm available tomorrow between 9:00 AM and 5:00 PM. Would you like to schedule a meeting?
```

### Available Endpoints

- `POST /chat`: Process a new message
- `GET /threads`: List all conversation threads
- `GET /threads/{thread_id}`: Get conversation history for a thread

## 🧪 Testing

Run the test suite:
```bash
poetry run pytest
```

Run with coverage:
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

## 📚 Documentation

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Calendar API](https://developers.google.com/calendar/api)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Poetry](https://python-poetry.org/)
