
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging 
import pickle 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train_processed, y_train_encoded, x_test_processed, y_test_encoded, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i] 
            
            logging.info(f"Starting hyperparameter tuning for model: {model_name}") 

            if model_name == "Logistic Regression":
                current_model_params = []
            
                base_params = {k: v for k, v in param[model_name].items() if k not in ['penalty', 'l1_ratio']}
                
                for penalty_val in param[model_name].get('penalty', []):
                    if penalty_val == 'elasticnet':
                        
                        for l1_ratio_val in param[model_name].get('l1_ratio', []):
                            temp_params = {'penalty': [penalty_val], 'l1_ratio': [l1_ratio_val]}
                            temp_params.update(base_params)
                            current_model_params.append(temp_params)
                    elif penalty_val is None: 
                        temp_params = {'penalty': [None]}
                        temp_params.update(base_params)
                        current_model_params.append(temp_params)
                    else: 
                        temp_params = {'penalty': [penalty_val]}
                        temp_params.update(base_params)
                        current_model_params.append(temp_params)
                
                if not current_model_params:
                    current_model_params = [base_params] 
            else:
                current_model_params = param[model_name]

            
            gs = RandomizedSearchCV(model, current_model_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
            gs.fit(x_train_processed, y_train_encoded)

            model.set_params(**gs.best_params_)
            model.fit(x_train_processed, y_train_encoded) # Retrain with best parameters

            y_train_pred = model.predict(x_train_processed)
            y_test_pred = model.predict(x_test_processed)

            
            train_model_score = accuracy_score(y_train_encoded, y_train_pred)
            test_model_score = accuracy_score(y_test_encoded, y_test_pred)

            logging.info(f"Finished tuning {model_name}.")
            logging.info(f"Train Accuracy: {train_model_score:.4f}")
            logging.info(f"Test Accuracy: {test_model_score:.4f}")
            logging.info("-" * 50) 

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)