## AI Text Summarizer

## Overview
This project implements an AI-driven text summarization tool with a FastAPI backend and Streamlit frontend.


## Features
- Text summarization with configurable length (short, medium, long)
- Multiple summarization methods (Sumy, HuggingFace, OpenAI)
- Summary history storage and retrieval
- User-friendly Streamlit interface

## Project Structure
```
text_summarization_ai/
│-- backend/
│   ├── app.py          # FastAPI backend
│   ├── database.py     # Database interactions
│   ├── text_summarization.py  # OpenAI integration
│   ├── requirements.txt # Dependencies
│   ├── Dockerfile      # Docker deployment
│-- frontend/
│   ├── app.py          # Streamlit frontend
│   ├── Dockerfile      # Docker deployment
│-- .env                # API keys (excluded from version control)
│-- README.md           # Project documentation
│-- docker-compose.yml  # Deployment config
```

## Prerequisites
- Python 3.9+
- OpenAI API key
- Docker and Docker Compose (for containerized deployment)


## Setup and Installation

### Clone the repository:
```sh
git clone https://github.com/HarshiniGudipally/text_summarization_ai.git
cd text_summarization_ai
```

### Set up a virtual environment:
```sh
python -m ai_text_summarizer
source ai_text_summarizer/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install dependencies:
```sh
pip install -r backend/requirements.txt
pip install streamlit
```

### Set up MongoDB:
- Install MongoDB and ensure it's running on localhost:27017
- Update the connection string in backend/database.py if needed

### Configure API keys:
- Add your OpenAI API key to backend/text_summarization.py
- Create a `.env` file in the project root and add your OpenAI API key:
```sh
OPENAI_API_KEY="your_api_key_here"
```
### Start the backend:
```sh
uvicorn backend.app:app --reload
```

### Start the frontend:
```sh
streamlit run frontend/app.py
```

### Access the application at [http://localhost:8501](http://localhost:8501)

### Usage
- Enter text in the provided text area
- Select summary length and summarization method
- Click "Summarize" to generate a summary
- View recent summaries in the history section

## Acknowledgments
- OpenAI for providing the GPT model
- FastAPI and Streamlit for the web framework and UI
- Motor for database operations
