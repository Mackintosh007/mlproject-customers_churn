                                 churn_prediction_project
🚀 Customer Churn Prediction: An End-to-End MLOps Project
This repository contains an end-to-end Machine Learning Operations (MLOps) project designed to predict customer churn. The goal is to identify customers at risk of churning, allowing businesses to proactively intervene and improve customer retention.

✨ Motivation
Customer churn is a critical challenge for businesses, directly impacting revenue and growth. By leveraging machine learning, this project aims to build a robust system that can:

Predict Churn: Identify customers likely to churn based on their behavioral and demographic data.

Provide Insights: Offer actionable insights into the factors contributing to churn.

Support Business Decisions: Enable proactive retention strategies and targeted interventions.

🛠️ Technologies & Features
This project utilizes a modern Python-based tech stack, incorporating best practices for data science and MLOps:

Python 3.9+: The core programming language.

Pandas & NumPy: For efficient data manipulation and numerical operations.

Scikit-learn: For data preprocessing (Imputation, Scaling, One-Hot Encoding) and core machine learning algorithms (Logistic Regression, K-Neighbors Classifier, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Support Vector Classifier).

CatBoost & XGBoost: High-performance gradient boosting frameworks for robust predictive modeling.

Flask: A lightweight web framework for building the prediction API and front-end.

dill & pickle: For serialization and deserialization of Python objects (models, preprocessors).

Logging: Integrated for tracking application flow and debugging.

HTML & Tailwind CSS: For a responsive and user-friendly web interface.

Power BI (Conceptual): The project's analytical approach is inspired by real-world business intelligence needs, as demonstrated in a PwC Switzerland Power BI Job Simulation.


📦 Setup & Installation
Follow these steps to set up the project locally:

Clone the Repository:

git clone https://github.com/your_username/churn_prediction_project.git
cd churn_prediction_project

Create and Activate Conda Environment:
It's highly recommended to use a Conda environment to manage dependencies, especially for data science projects.

conda create --name venv python=3.9 numpy pandas # Create environment with core libraries
conda activate venv                             # Activate the environment

Install Required Libraries:
Once the environment is active, install all necessary Python packages.

pip install Flask scikit-learn catboost xgboost dill

Install Visual Studio Build Tools (Windows Only):
If you encounter compilation errors during pip install (e.g., related to vswhere.exe), you need to install the Microsoft Visual C++ Build Tools.

Download from Visual Studio Downloads.

Under "Tools for Visual Studio," download "Build Tools for Visual Studio 2022" (or latest).

Run the installer and ensure to select the "Desktop development with C++" workload.

Restart your computer after installation.

🏃 How to Run the Application
1. Train the Model & Generate Artifacts
First, you need to run the data pipeline to train the model and save the model.pkl and preprocessor.pkl artifacts.

python -m src.components.data_ingestion

This script will:

Ingest data.

Transform data.

Train various ML models.

Evaluate models and select the best one.

Save the best model.pkl and the fitted preprocessor.pkl into the artifacts/ directory.

2. Start the Web Application
After the artifacts are generated, you can start the Flask web server:

Ensure your Conda environment venv is active.

Navigate to the project root directory (where app.py is located).

Run the Flask application:

python app.py

You should see output indicating the server is running, like:
* Running on http://127.0.0.1:25565 (the port might vary).

3. Access the Web Interface
Open your web browser.

Navigate to the URL displayed in your terminal (e.g., http://127.0.0.1:25565).

You will see a Customer Churn Prediction form. Fill in the customer details and click "Predict Churn."

The prediction (e.g., "Yes (Customer will churn)" or "No (Customer will not churn)") will be displayed on the page.

📈 Model Performance
The best-performing model achieved an accuracy of 0.868% on the test set.

🛠️ Future Enhancements
API Documentation: Implement OpenAPI/Swagger for better API documentation.

Containerization: Dockerize the application for easier deployment across environments.

CI/CD Pipeline: Set up a Continuous Integration/Continuous Deployment pipeline (e.g., with GitHub Actions, Jenkins) for automated testing and deployment.

Monitoring: Implement monitoring for model performance drift and data quality.

More Advanced Models: Experiment with deep learning models for churn prediction.

Feature Engineering: Explore more sophisticated feature engineering techniques.

✒️ Author
Mackintosh Nnamdi Odika