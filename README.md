# Fraud Detection System 🚦💳

An end-to-end machine-learning pipeline that **scores real-time financial transactions for fraud**.  
It combines classic ML models (Logistic Regression & Random Forest) with a lightweight Flask REST API, is fully containerised with Docker, and is production-ready for AWS ECS (Fargate) behind an Application Load Balancer. 

---

## Table of Contents
1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Quick Start](#quick-start)  
4. [API Reference](#api-reference)  
5. [Retraining / Experiments](#retraining--experiments)  
6. [Deploying to AWS ECS](#deploying-to-aws-ecs)  
7. [Project Structure](#project-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Features
| Category | Details |
|----------|---------|
| **Models** | Logistic Regression & Random Forest (sklearn) persisted as `.pkl` artifacts |
| **Pre-processing** | Custom `feature_transformer.py` handles scaling & PCA-like anonymised components |
| **API** | Flask app (`app.py`) exposes `/predict`, `/healthz`, `/version` endpoints |
| **Containerisation** | 17-line `Dockerfile` based on Python 3.11 slim, tini, non-root user |
| **CI/CD-ready** | Clean `requirements.txt`; opinionated for GitHub Actions & Docker Hub / ECR pushes |
| **Cloud** | Reference Terraform/ECS instructions (see below) for blue-green deployments |
| **Data** | Trained on the public Kaggle “Credit Card Fraud Detection” dataset (284 K txns, 492 fraud) |

---

## Architecture

```text
                          ┌──────────────┐
        Docker image ⇢    │  Flask API   │  ⇠ Requests (JSON)
   (LogReg / RF models)   └──────┬───────┘
                                 │ in-container Unix socket
                                 ▼
         ┌──────────────────────────────────┐
         │  Model Service (scikit-learn)    │
         │  • load .pkl at start-up         │
         │  • feature_transformer           │
         │  • predict_proba & threshold     │
         └──────────────────────────────────┘
                                 │
                          ┌──────┴───────┐
                          │  AWS ALB     │
                          └──────┬───────┘
                                 │
                          ┌──────┴───────┐
                          │  ECS Service │
                          └──────────────┘


## Quick Start

### 1 · Clone & set up a virtual environment
```bash
git clone https://github.com/WenjingDong/Fraud_Detection_System.git
cd Fraud_Detection_System

# (Optional) Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
