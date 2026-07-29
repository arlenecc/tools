# OpenAI Debug Tool

A PyQt6-based GUI application for debugging OpenAI-compatible API services.

## Features

- **Configuration Panel**
  - Set Base URL (e.g., `https://api.openai.com/v1`)
  - Set API Key
  - Select or enter Model name
  - Fetch available models with one click
  - Save/Load configuration persistently

- **Chat Interface**
  - Interactive conversation display
  - Streaming responses with real-time updates
  - Token generation speed display (tokens/second)
  - Clear chat history option

- **Debug Logging**
  - Real-time log panel showing application events
  - Timestamped entries with log levels
  - Clear logs option

## Installation

```bash
pip install pyqt6 requests
```

## Usage

Run the application:

```bash
python run.py
```

Or directly:

```bash
python src/main.py
```

**Note for headless environments (e.g., Docker, WSL without display):**

Set the Qt platform to offscreen mode:

```bash
QT_QPA_PLATFORM=offscreen python run.py
```

This allows the application to run without a physical display, useful for testing or remote servers.

## Project Structure

```
/workspace
├── run.py                 # Application entry point
├── src/
│   ├── main.py           # GUI application (PyQt6)
│   ├── config_manager.py # Configuration management
│   ├── api_client.py     # OpenAI API client
│   ├── message_history.py# Conversation history
│   ├── speed_calculator.py# Token speed calculation
│   ├── logger.py         # Debug logging
│   └── log_entry.py      # Log entry class
├── tests/
│   ├── test_openai_debug_tool.py  # Unit tests for core modules
│   └── test_gui.py       # Tests for GUI components
└── README.md
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

## API Compatibility

This tool works with any OpenAI-compatible API endpoint, including:
- OpenAI API (`https://api.openai.com/v1`)
- Local LLM servers (Ollama, LM Studio, etc.)
- Custom API implementations following OpenAI spec

## License

MIT License
