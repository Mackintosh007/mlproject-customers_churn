import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score # For evaluation
from sklearn.linear_model import LogisticRegression # For linear classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from catboost import CatBoostRegressor # parameters for classification
from xgboost import XGBClassifier # Convenience class for XGBoost classification

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

CATBOOST_LOG_DIR = os.path.join('artifacts', 'catboost_logs')
os.makedirs(CATBOOST_LOG_DIR, exist_ok=True)

CLASSIFICATION_SCORING_METRIC = "accuracy_score"

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train_processed, y_train_encoded, x_test_processed, y_test_encoded = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
                "Support Vector Classifier": SVC(random_state=42), 
                "CatBoost Classifier": CatBoostClassifier(random_state=42, train_dir=CATBOOST_LOG_DIR, verbose=0),
                "XGB Classifier": XGBClassifier(random_state=42)
            }
            params ={
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.1, 1], 
                    'solver': ['saga'],
                    'max_iter': [2000], 
                    'l1_ratio': [0.5] 
                },
                    
                "K-Neighbors Classifier": {
                    'n_neighbors': [5],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },

                "Random Forest Classifier": {
                    'n_estimators': [100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },

                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)] # Use shallow trees
                },

                "Gradient Boosting Classifier": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                },

                "Support Vector Classifier": {
                    'C': [1],
                    'kernel': ['rbf', 'sigmoid'],
                    'gamma': ['scale']
                },

                "CatBoost Classifier": {
                    'loss_function': ['Logloss'], 
                    'iterations': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7]
                    
                },

                "XGB Classifier": {
                    'objective': ['binary:logistic'], 
                    'n_estimators': [100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3],
                    'subsample': [1.0],
                    'colsample_bytree': [1.0],
                    'gamma': [0, 0.2],
                    'reg_alpha': [0, 0.005, 0.01, 0.1],
                    'reg_lambda': [0, 0.005, 0.01, 0.1]
                },
            }

            model_report: dict = evaluate_models(
                x_train_processed=x_train_processed, 
                y_train_encoded=y_train_encoded,     
                x_test_processed=x_test_processed,   
                y_test_encoded=y_test_encoded,       
                models=models,
                param=params
            )
            
            ## To get best model score from dict
            best_model_score = max(model_report.values())

            ## To get best model name from dict

            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing datasetBest {CLASSIFICATION_SCORING_METRIC} is {best_model_score:.4f}.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test_processed)
            final_score = accuracy_score(y_test_encoded, predicted)
            return final_score
            



            
        except Exception as e:
            raise CustomException(e,sys)