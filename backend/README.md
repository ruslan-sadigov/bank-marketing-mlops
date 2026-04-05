# Bank Marketing MLOps - Backend API

A FastAPI backend service for bank term deposit subscription predictions with async file processing and comprehensive error handling.

## Features

- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Multi-format Support**: CSV, Excel (.xlsx, .xls) file processing
- **Async Processing**: Non-blocking file uploads and processing
- **CORS Enabled**: Cross-origin resource sharing for frontend integration
- **Health Monitoring**: Built-in health check endpoint
- **Timezone Support**: UTC and Baku timezone tracking
- **Error Handling**: Comprehensive exception handling and logging
- **Auto Documentation**: Interactive API docs via Swagger UI
- **Container Ready**: Fully dockerized with UV dependency management

## Technology Stack

- **Web Framework**: FastAPI 0.100+
- **Async Runtime**: Uvicorn ASGI server
- **Data Processing**: Custom ML prediction pipeline
- **File Handling**: Async file upload processing
- **Logging**: Python logging with structured output
- **Dependency Management**: UV (ultra-fast Python package installer)
- **Containerization**: Docker build


## Architecture

### Project Structure
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for exploratory data analysis and model experiments.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── uv.lock   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `uv lock > uv.lock`
    │
    ├── pyptoject.toml    <- makes project uv installable (uv installs) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── app.py                    # FastAPI application entry point
    ├── Dockerfile                # Container build instructions
    ├── pyproject.toml            # UV project configuration & dependencies
    ├── uv.lock                   # Locked dependency versions
    └── README.md                 # This file

--------


### Key Components

1. **FastAPI Application**: Main web server with CORS middleware
2. **Prediction Pipeline**: ML model inference via `src.models.predict_model`
3. **File Processor**: Async file upload and validation
4. **Health Monitor**: System status and timezone tracking
5. **Error Handler**: Comprehensive exception management
6. **Logger**: Structured logging for debugging and monitoring

## Docker Setup

This backend API is fully containerized and designed to work as part of a Docker Compose stack.

## Quick Start

### Option 1: Docker Compose (Recommended)

The backend is part of a larger Docker Compose application:

```bash
# Run the entire stack
docker compose up -d

# View logs
docker compose logs backend

# Stop the stack
docker compose down
```

The API will be available at `http://localhost:8000`

### Option 2: Standalone Docker

```bash
# Build the backend image (run from project root)
docker build -t bank-marketing-backend -f backend/Dockerfile .

# Run the container
docker run -p 8000:8000 bank-marketing-backend
```

### Option 3: Local Development

For local development without Docker:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the application
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check

**GET** `/health`

Returns system status and timezone information.

**Response:**
```json
{
  "status": "healthy",
  "utc_time": "2024-08-14T10:30:00.000000+00:00",
  "baku_time": "2024-08-14T14:30:00.000000+04:00"
}
```

### Make Predictions

**POST** `/predict`

Upload a file and receive ML predictions.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with file upload
  - `file`: CSV or Excel file (.csv, .xlsx, .xls)

**Response:**
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

**Error Response:**
```json
{
  "detail": "Invalid file format. Please upload CSV or Excel files only."
}
```

### Interactive API Documentation

- **Swagger UI**: `http://localhost:8000/docs`

## Configuration

### Docker Compose Integration

```yaml
# Example docker compose.yml snippet
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    networks:
      - app_network
    restart: always
    image: backend
    container_name: backend
    volumes:
      - ./backend:/app
      - backend_venv:/app/.venv
    ports:
      - "8000:8000"
```

### CORS Configuration

The API is configured to accept requests from any origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
