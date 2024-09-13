# CI/CD Pipeline for Machine Learning (Igbo NLP) using DVC, Docker, CML, AWS, Flask API, and GITHUB Actions

## Project Overview
This repository demonstrates the implementation of a CI/CD pipeline for a machine learning project. The pipeline automates key steps in the ML lifecycle, including data injesttion, data preprocessing, hyperparameter tuning, model training, evaluation, and deployment to AWS ECS, using Docker and GPU acceleration. We use modern tools like CML, DVC, and GitHub Actions to orchestrate the process.

By adopting this pipeline, we ensure that the machine learning model is consistently updated and retrained when there are data or code changes, providing an efficient and scalable solution for continuous integration and delivery in machine learning projects.

## Features

#### Data Versioning with DVC:
- Automatically tracks changes in datasets, triggering retraining when new data is added or existing data is modified.

### Model Training and Evaluation:
- Utilizes **CML** to train the machine learning model, followed by hyperparameter tuning using **Optuna**.
- Automatically generates metrics like accuracy, precision, recall, and F1 score.

### Docker for Consistent Environments:
- Ensures consistent environments across development, testing, and production by containerizing the project with Docker.
- Supports **GPU acceleration** for faster model training.

### Flask API for Model Inference:
- A **Flask API** is implemented to serve the trained model for real-time inference.
- The API allows users to send HTTP POST requests with text inputs, which are preprocessed using the same pipeline used during training.
- The API responds with the sentiment predictions, along with probabilities and class labels.

### Deployment on AWS:
- The trained model is deployed to **AWS ECS** (Elastic Container Service), ensuring scalable and highly available inference.

## Technologies Used

- **Python**: For model development and scripting.
- **Flask**: To serve the trained machine learning model via a RESTful API.
- **Docker**: For containerizing the machine learning environment.
- **GitHub Actions**: Automates CI/CD workflows, including building, testing, and deployment.
- **CML (Continuous Machine Learning)**: Automates the machine learning workflow.
- **DVC (Data Version Control)**: Tracks and versions datasets and models.
- **AWS ECS**: Deploys the Dockerized machine learning model for scalable inference.
