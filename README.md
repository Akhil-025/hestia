# Hestia

A local-first personal AI system that unifies your documents, memory, and photos into a single assistant.
Think of it as a private AI assistant that can remember what you tell it, read your notes, and understand your photos — all on your own machine.

---

## What Hestia Does

- Chat with your personal documents (RAG)
- Remember facts, preferences, and past conversations
- Search and understand your photos using AI captions
- Run everything locally with full data privacy
---

## Why Hestia?

Unlike most AI tools that focus on a single domain (documents, memory, or media),
Hestia unifies all three into one system.

- Combines memory + documents + images in a single query
- Designed to run locally (optional cloud integrations supported)
- Modular architecture for extensibility

---

## Key Features

- Natural language query routing across all modules
- Question answering from ingested personal documents (RAG via Athena)
- Long-term memory of user facts, preferences, and past interactions (Mnemosyne)
- Photo and media search with automatic image captioning and tag-based retrieval (Iris)
- Web UI (Flask) for document and image ingestion, search, and interaction
- Telegram bot integration for remote queries and commands
- Voice input and output (if configured: Whisper for STT, pyttsx3 for TTS)
- Configurable modules and storage locations

---

## System Architecture

- **Hestia Core**: Orchestration, natural language understanding (NLU), intent routing, memory management, and action execution.
- **Athena**: Retrieval-Augmented Generation (RAG) system for document intelligence. Uses ChromaDB for semantic search and context retrieval.
- **Mnemosyne**: Long-term memory subsystem. Stores user facts, preferences, and conversation history in SQLite and ChromaDB.
- **Iris**: Media intelligence subsystem. Performs image captioning and tag-based search using LLaVA (via Ollama) and manages a local image database.

---

## Motivation

Most AI tools are fragmented — one for documents, one for notes, one for media.
Hestia brings these capabilities together into a single system that runs locally,
retains memory, and can reason across your personal data.

---

## How It Works

1. **Input**: User submits a query (via CLI, web UI, Telegram, or voice).
2. **NLU**: The system parses the query to determine intent and extract entities.
3. **Routing**: The query is routed to the appropriate module (Athena, Mnemosyne, Iris, or core actions).
4. **Module Processing**: The selected module processes the request (e.g., document search, memory lookup, image analysis).
5. **Response**: The system returns a structured answer via the chosen interface.

---

## Data Flow

- **SQLite**: Used by Mnemosyne for long-term memory and by Iris for image metadata.
- **ChromaDB**: Used by Athena for document embeddings and semantic search; also used by Mnemosyne for memory embeddings.
- **Iris DB**: Stores image records, captions, tags, and analysis results.
- **Data directories**: All user data is stored locally under the data directory, organized by module.

---

## Configuration

- Main configuration file: laptop_config.yaml
  - Set paths, enable/disable modules, configure model endpoints, and adjust system parameters.
- Module-specific settings are also loaded from the config directory.
- To enable or disable modules, adjust the relevant flags in the YAML config.

---

## Example Queries

- **Memory**
  - "do you remember my favorite color?"
  - "what did I tell you yesterday?"
  - "what do you know about me?"
- **Documents**
  - "explain entropy from my notes"
  - "find the summary of the project proposal"
  - "search for 'machine learning' in my documents"
- **Images**
  - "find photos where someone is reading"
  - "show me pictures from last summer"
  - "describe the latest image"
- **System/Commands**
  - "ingest new documents"
  - "analyze new photos"
  - "reset memory"
  - "help"

---

## Cross-Module Capabilities

Hestia is designed to combine information across modules:

- "search my notes and explain it simply" → Athena + Core LLM
- "what do you know about me" → Mnemosyne
- "find photos of people reading" → Iris
- "remember that I prefer dark mode" → Mnemosyne + Core

Future capability:
- "compare what I studied with what I experienced" (documents + memory + media)

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/hestia.git
   cd hestia
   ```
2. **Set up Python environment**
   ```sh
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\\Scripts\\activate
   ```
3. **Install Python requirements**
   ```sh
   pip install -r requirements.txt
   ```
4. **Install Ollama**
   - Download and install from [https://ollama.com/](https://ollama.com/)
5. **Pull required models**
   ```sh
   ollama pull llava:7b
   ollama pull mistral
   # Add other models as needed
   ```

---

## Running the System

- **CLI Mode**
  ```sh
  python main.py
  ```
- **Web UI (Flask)**
  ```sh
  python web_ui.py
  ```
  - Access via browser at `http://localhost:5000`
- **Telegram Bot**
  - Configure your Telegram token in the config file.
  - Start the bot:
    ```sh
    python -m core.telegram_bot
    ```
- **Voice Mode**
  - If enabled in config, voice input/output is available via CLI or web UI.

---

## Workflows

- **Document Ingestion (Athena)**
  - Place documents in `data/athena/documents/`
  - Use the web UI or CLI to trigger ingestion and indexing.
- **Memory Storage (Mnemosyne)**
  - User facts and interactions are stored automatically during conversation.
- **Image Analysis (Iris)**
  - Place images in your configured source directory (default: `C:/Users/<user>/Pictures`)
  - Use the web UI or CLI to trigger analysis and captioning.

---

## Project Structure

```
hestia/
  core/                # orchestration, NLU, memory, actions, Telegram, web UI
  modules/
    athena/            # RAG/document intelligence
    mnemosyne/         # long-term memory
    iris/              # photo/media intelligence
  data/
    athena/
    mnemosyne/
    iris/
  config/
    laptop_config.yaml
  main.py
  web_ui.py
  requirements.txt
```

---

## Current Status

- Athena: working (document ingestion, semantic search, RAG)
- Mnemosyne: working (long-term memory, fact storage, recall)
- Iris: working (image captioning, tag-based search)
- Web UI: working (basic features)
- Telegram bot: working
- Voice: working (if configured)
- Some advanced features are experimental or in progress

---

## Roadmap

- Background processing (auto ingestion and analysis)
- Semantic image search (CLIP-based similarity)
- Improved query routing (NLU-driven, less heuristic)
- Multi-device memory synchronization

---

## Philosophy

Hestia is designed to be local-first, modular, and privacy-respecting. All data and models run on your own machine. The system is built for extensibility and transparency, not as a cloud service.

## Limitations

- Image search is currently based on captions and tags, not visual embeddings
- Requires local model setup (Ollama), which may be resource-intensive
- Query routing relies on heuristic triggers and NLU confidence
- Multi-device sync is not yet implemented


## Maintainer

Akhil Pillai  
Mechanical Engineering @ SPCE  
Interests: AI systems, robotics, thermal systems