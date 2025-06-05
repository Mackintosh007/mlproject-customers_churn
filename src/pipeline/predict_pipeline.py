import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: int,
        TotalCharges: int,
        numAdminTickets: int,
        numTechTickets: int
        ):

        self.gender = gender

        self.SeniorCitizen = SeniorCitizen

        self.Partner = Partner

        self.Dependents = Dependents

        self.tenure = tenure

        self.PhoneService = PhoneService

        self.MultipleLines = MultipleLines

        self.InternetService = InternetService

        self.OnlineSecurity = OnlineSecurity

        self.StreamingTV = StreamingTV

        self.StreamingMovies = StreamingMovies

        self.Contract = Contract

        self.PaperlessBilling = PaperlessBilling

        self.PaymentMethod = PaymentMethod

        self.MonthlyCharges = MonthlyCharges

        self.TotalCharges = TotalCharges

        self.numAdminTickets = numAdminTickets

        self.numTechTickets = numTechTickets



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "gender": [self.gender],
            "SeniorCitizen": [self.SeniorCitizen],
            "Partner": [self.Partner],
            "Dependents": [self.Dependents],
            "tenure": [self.tenure],
            "PhoneService": [self.PhoneService],
            "MultipleLines": [self.MultipleLines],
            "InternetService": [self.InternetService],
            "OnlineSecurity": [self.OnlineSecurity],
            "StreamingTV": [self.StreamingTV],
            "StreamingMovies": [self.StreamingMovies],
            "Contract": [self.Contract],
            "PaperlessBilling": [self.PaperlessBilling],
            "PaymentMethod": [self.PaymentMethod],
            "MonthlyCharges": [self.MonthlyCharges],
            "TotalCharges": [self.TotalCharges],
            "numAdminTickets": [self.numAdminTickets],
            "numTechTickets": [self.numTechTickets],
        }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

