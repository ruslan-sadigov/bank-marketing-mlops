# Bank Marketing Campaign Prediction Model

![Build Status](https://github.com/HuGoBass09/bank-marketing-mlops/actions/workflows/ci-build.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://shields.io/)

---

## Overview

An MLOps project that predicts whether a bank customer will subscribe to a term deposit. It consists of a FastAPI backend serving an XGBoost model and a Streamlit frontend for uploading customer data and viewing predictions. Both services are containerized with Docker and orchestrated via Docker Compose.

## Architecture

- **Backend**: FastAPI serving the XGBoost model via `/predict` endpoint
- **Frontend**: Streamlit UI for file upload and prediction results
- **Containerization**: Docker and Docker Compose for service orchestration
- **CI**: GitHub Actions with Ruff linting on every push/PR

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git

### Running the Application

1. **Clone the repository**
   ```bash
   git clone https://github.com/HuGoBass09/bank-marketing-mlops.git
   cd bank-marketing-mlops
   ```

2. **Start services**
   ```bash
   # Build and start both services in the background
   docker compose up -d

   # View logs
   docker compose logs -f

   # Stop services
   docker compose down
   ```

3. **Access the application**
   - Frontend: `http://localhost:8501`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## Project Structure

```
bank-marketing-deployment/
├── backend/              # FastAPI application and ML pipeline
├── frontend/             # Streamlit application
├── docker-compose.yml    # Service orchestration
├── .github/workflows/    # CI pipeline
└── README.md
```

## License

Licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
