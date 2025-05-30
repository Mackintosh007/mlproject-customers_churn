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

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
                "Support Vector Classifier": SVC(random_state=42), 
                "CatBoost Classifier": CatBoostClassifier(random_state=42, train_dir=CATBOOST_LOG_DIR),
                "XGB Classifier": XGBClassifier(random_state=42)
            }
            params ={
                "logistic_regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['saga'], # 'saga' supports 'l1', 'l2', 'elasticnet', 'none'
                    'max_iter': [1000]
                },
                    
                "kneighbors_classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                "decision_tree_classifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },

                "random_forest_classifier": {
                    'n_estimators': [100, 200, 300],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },

                "ada_boost_classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)] # Use shallow trees
                },

                "gradient_boosting_classifier": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                },

                "svc": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                    # 'degree': [2, 3, 4] # Only relevant if kernel='poly'
                },

                "catboost_classifier": {
                    'loss_function': ['Logloss'], # Use ['Logloss'] for binary classification
                    # 'loss_function': ['MultiClass'], # Use ['MultiClass'] for multi-class classification
                    'iterations': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7]
                    # 'random_seed': [42]
                },

                "xgboost_classifier_params": {
                    'objective': ['binary:logistic'], # Use ['binary:logistic'] for binary classification
                    # 'objective': ['multi:softmax'], # Use ['multi:softmax'] for multi-class classification
                    'n_estimators': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'reg_alpha': [0, 0.005, 0.01, 0.1],
                    'reg_lambda': [0, 0.005, 0.01, 0.1]
                },
            }

            model_report:dict = evaluate_models(
                x_train = x_train_processed,
                y_train = y_train_encoded,
                X_test = x_test_processed,
                y_test = y_test_encoded,
                models = models,
                param = params
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

            predicted = final_best_model.predict(x_test_processed)
            final_score = accuracy_score(y_test_encoded, predicted)
            return accuracy_score
            



            
        except Exception as e:
            raise CustomException(e,sys)