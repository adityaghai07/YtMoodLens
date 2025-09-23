# YtMoodLens

A real-time YouTube sentiment analysis tool that leverages MLOps principles to analyze the emotional tone of YouTube comments through a browser extension.



https://github.com/user-attachments/assets/26715a49-6f11-4529-b7b8-b7f10d2d3a14


---


## Model Training Flow

```mermaid
graph TD
    A[Raw Data] --> B[Feature Engineering]
    B --> C[Data Preprocessing]
    C --> D[Train/Test Split]
    
    D --> E[Model Training Pipeline]
    E --> F[Logistic Regression]
    E --> G[LGBM]
    E --> H[Other Models]
    
    F --> I[Hyperparameter Tuning]
    G --> I
    H --> I
    
    I --> J[Handle Class Imbalance]
    J --> K[Better Vectorizers]
    K --> L[Model Evaluation]
    
    L --> M[Metrics Comparison]
    M --> N{Best Model Selection}
    
    N --> O[Register Best Model in MLflow]
    O --> P[Model URI Generated]
    
    P --> Q[S3 Storage]
    Q --> R[EC2 Deployment Instance]
    R --> S[Load Model via URI]
    S --> T[Production Model]
    
    subgraph "MLflow Tracking"
        U[Experiment Tracking]
        V[Metrics Logging]
        W[Model Registry]
    end
    
    subgraph "AWS Infrastructure"
        Q
        R
        X[EC2 Training Instance]
    end
    
    E --> U
    L --> V
    O --> W
    E -.-> X

```


--- 

## Overview

YtMoodLens targets MLOps principles by implementing a complete machine learning pipeline that trains models with improved methodologies through systematic experiments. The models are stored in S3 via MLflow for version control and model management, then loaded and served on EC2 instances for real-time inference.

## Architecture

- **Model Training & Experimentation**: Advanced training methods with experiment tracking with mlflow
- **Model Storage**: Models stored in S3 using MLflow for versioning and artifact management
- **Container Registry**: ECR (Elastic Container Registry) for Docker image management
- **Deployment**: Served on EC2 instances for scalable inference
- **Data Source**: YouTube API from GCP for comment extraction
- **Frontend**: Browser extension providing near real-time sentiment analysis

---

## Flow Diagram


``` mermaid
graph TD
    A[Chrome Plugin] --> B[YouTube API Request]
    B --> C[GCP YouTube Data API v3]
    C --> D[Comments Data]
    
    D --> E[Plugin Frontend]
    E --> F[API Call to Backend]
    
    F --> G[Hosted Model API]
    G --> H[ML Predictions]
    H --> I[Generate Charts]
    H --> J[Generate Word Cloud]
    
    I --> K[Return Results to Plugin]
    J --> K
    
    subgraph "Backend CI/CD Pipeline"
        L[GitHub Repository] --> M[GitHub Actions]
        M --> N[Build Docker Image]
        N --> O[Push to AWS ECR]
        O --> P[Deploy to EC2]
    end
    
    subgraph "AWS Infrastructure"
        Q[ECR Registry]
        R[EC2 Instance]
        S[Model API Endpoint]
    end
    
    subgraph "GCP Services"
        C
        T[API Key Management]
    end
    
    N --> Q
    Q --> R
    R --> S
    G --> S
    
    C --> T
    B --> T
    
    P --> U[Production Backend]
    U --> G
    
    K --> V[Display in Chrome Plugin]
    V --> W[Charts & Word Cloud UI]
    V --> X[Prediction Results]

```

---


## Features

- Real-time sentiment analysis of YouTube comments
- MLOps-driven model development and deployment pipeline
- Scalable cloud infrastructure on AWS
- Browser extension for seamless user experience
- Integration with Google Cloud Platform's YouTube API

## Technology Stack

- **Cloud Platform**: AWS (EC2, S3, ECR)
- **ML Platform**: MLflow for experiment tracking and model management, DVC for versioning
- **API**: YouTube Data API v3 (Google Cloud Platform)
- **Frontend**: Browser extension (Chrome/Firefox compatible)
- **Containerization**: Docker with ECR


## Special Thanks

Special thanks to [dswithbappy](https://github.com/entbappy) for this project.
