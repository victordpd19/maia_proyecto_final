# Invoice Data Extraction Service

This project consists of a FastAPI backend service for extracting data from PDF invoices and a Streamlit frontend for user interaction.

## Features

- Extract vendor name, items, quantities, total tax, and date from PDF invoices
- OCR processing using Tesseract
- AI-powered data extraction using OpenAI GPT-4
- Real-time progress tracking
- Simple and intuitive user interface

## Project Structure

```
.
├── backend/                # FastAPI backend service
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   └── services/       # Business logic
│   ├── Dockerfile          # Docker configuration for backend
│   └── requirements.txt    # Python dependencies for backend
├── frontend/               # Streamlit frontend
│   ├── app.py              # Streamlit application
│   ├── Dockerfile          # Docker configuration for frontend
│   └── requirements.txt    # Python dependencies for frontend
├── docker-compose.yml      # Docker Compose configuration
├── .env.example            # Example environment variables
└── README.md               # Project documentation
```

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Tesseract OCR (installed automatically in Docker)

## Setup

1. Clone the repository
2. Copy the example environment file and add your OpenAI API key:
   ```
   cp .env.example .env
   ```
3. Edit the `.env` file and add your OpenAI API key

## Running the Application

### Using Docker Compose (recommended)

```bash
docker-compose up --build
```

This will start both the backend and frontend services. The frontend will be available at http://localhost:8501 and the backend API at http://localhost:8000.

### Running Locally (Development)

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Endpoints

- `GET /ping`: Health check endpoint
- `POST /extraction/extract`: Start an extraction task
- `GET /extraction/status/{extraction_id}`: Get the status of an extraction task
- `GET /extraction/results/{extraction_id}`: Get the results of a completed extraction task

## Deployment to Railway

This project is configured to be easily deployed to Railway:

1. Create a new project in Railway
2. Connect your GitHub repository
3. Add the required environment variables (OPENAI_API_KEY)
4. Deploy the service

## License

MIT 
