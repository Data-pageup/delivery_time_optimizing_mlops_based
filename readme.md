# Food Delivery Time Prediction MLOps Project ( in progress - an issuse with accessing real_time data)

## Project Overview

This project aims to predict the delivery time of food orders using Machine Learning and deploy the solution using MLOps best practices.

The model analyzes factors such as delivery partner information, traffic conditions, weather conditions, order details, and delivery distance to estimate the expected delivery time in minutes.

The project follows a complete MLOps lifecycle including:

* Data Exploration
* Data Preprocessing
* Feature Engineering
* Model Training
* Experiment Tracking with MLflow
* Model Deployment using FastAPI
* User Interface using Streamlit
* Docker Containerization
* CI/CD Automation
* Production Monitoring

---

## Business Problem

Food delivery platforms often struggle with inaccurate delivery estimates, leading to customer dissatisfaction and increased support requests.

The objective of this project is to build a machine learning system capable of predicting delivery time accurately so that:

* Customers receive realistic delivery estimates.
* Delivery operations can be optimized.
* Customer trust and retention can be improved.
* Businesses can proactively handle delays.

---

## Dataset

Source: Kaggle Food Delivery Dataset

The dataset contains information related to:

* Delivery Partner Age
* Delivery Partner Rating
* Restaurant Location
* Customer Location
* Weather Conditions
* Traffic Density
* Vehicle Condition
* Vehicle Type
* Order Type
* Festival Indicator
* City
* Multiple Deliveries
* Distance Between Restaurant and Customer

### Target Variable

```text
delivery_time_mins
```

---

## Project Structure

```text
food-delivery-mlops/

├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── artifacts/
├── mlruns/
├── notebooks/
├── tests/
│
├── app.py
├── streamlit_app.py
└── README.md
```

---

## Tech Stack

### Programming

* Python 3.10

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost

### MLOps

* MLflow
* Docker
* GitHub Actions

### Deployment

* FastAPI
* Streamlit

---

## Workflow

### Phase 1: Planning & Setup

* Define business problem
* Collect dataset
* Create project structure

### Phase 2: Experimentation

* Exploratory Data Analysis
* Data Preprocessing
* Feature Engineering
* Model Training

### Phase 3: Production Code

* Convert notebooks into reusable scripts
* Create training and inference pipelines

### Phase 4: Deployment

* FastAPI Backend
* Streamlit Frontend

### Phase 5: Containerization

* Dockerize application

### Phase 6: Testing

* Unit Testing
* API Testing

### Phase 7: CI/CD

* GitHub Actions Pipeline

### Phase 8: Cloud Deployment

* AWS / GCP / Azure

### Phase 9: Monitoring

* Performance Monitoring
* Model Retraining

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Create environment:

```bash
conda create -n delivery_time python=3.10
conda activate delivery_time
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

* Real-time ETA prediction
* Route optimization
* Drift detection
* Automated retraining pipeline
* Cloud-native deployment

---

## Author

Developed as an End-to-End MLOps Project for Food Delivery Time Prediction.
