# Marketing Campaign Predictor - Frontend

A Streamlit frontend for uploading customer data and generating term deposit subscription predictions via the backend ML service.

## Features

- **File Upload**: CSV and Excel (.xlsx, .xls) support
- **Data Preview**: Interactive preview and optional statistical summary
- **Predictions**: Sends data to backend API and displays probability scores and Yes/No labels
- **Decision Threshold**: Configurable slider (0.0–1.0) to tune the classification cutoff
- **Export**: Download predictions merged with original data as CSV
- **Session Management**: Uploaded file and results persist across interactions

## Technology Stack

- **Frontend Framework**: Streamlit 1.48.1+
- **Data Processing**: Pandas
- **HTTP Client**: Requests
- **Dependency Management**: UV
- **Containerization**: Docker

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start both frontend and backend services in the background
docker compose up -d

# Stream frontend logs (useful for debugging)
docker compose logs frontend

# Stop all services
docker compose down
```

The application will be available at `http://localhost:8501`

### Option 2: Local Development

```bash
# Install dependencies from pyproject.toml into .venv
uv sync

# Run the Streamlit app
uv run streamlit run app.py
```

## Architecture

### Project Structure

```
├── app.py              # Main Streamlit application
├── Dockerfile          # Container build instructions
├── pyproject.toml      # UV project configuration & dependencies
├── uv.lock             # Locked dependency versions
└── README.md           # This file
```

### Key Components

1. **File Upload Handler**: Validates and reads uploaded files into session state
2. **Data Processor**: Loads and previews uploaded data using pandas
3. **API Client**: Sends file to backend `/predict` endpoint and parses response
4. **Threshold Slider**: Applies configurable cutoff to probability scores client-side
5. **Results Display**: Merges predictions and probabilities with original data, enables CSV download

## Configuration

### Backend API URL

Configurable via the sidebar in the UI. Defaults to `http://backend:8000/predict` for Docker Compose. Change to `http://localhost:8000/predict` for local development without Docker.

### Expected Data Format

The uploaded file should contain the following columns:

| Column | Type |
|---|---|
| age | numerical |
| balance | numerical |
| day | numerical |
| campaign | numerical |
| pdays | numerical |
| previous | numerical |
| job | categorical |
| marital | categorical |
| education | categorical |
| default | categorical |
| housing | categorical |
| loan | categorical |
| contact | categorical |
| month | categorical |
| poutcome | categorical |

### Expected API Response

```json
{
  "status": "success",
  "message": "Predictions generated successfully",
  "data": {
    "predictions": [0, 1, 0, 1],
    "probabilities": [0.1428, 0.8732, 0.2341, 0.7651],
    "num_predictions": 4,
    "processing_time_seconds": 0.234
  }
}
```
