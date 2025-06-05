from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

# Configure logging to see messages in the console
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


application=Flask(__name__) 

app=application 


@app.route('/')
def index():
    
    return render_template('index.html') 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html') 
    else:
        logging.info("Received POST request for prediction.")
        try:
            data=CustomData(
                gender=request.form.get('gender'),
                SeniorCitizen=int(request.form.get('SeniorCitizen')), 
                Partner=request.form.get('Partner'),
                Dependents=request.form.get('Dependents'),
                tenure=int(request.form.get('tenure')), 
                PhoneService=request.form.get('PhoneService'),
                MultipleLines=request.form.get('MultipleLines'),
                InternetService=request.form.get('InternetService'),
                OnlineSecurity=request.form.get('OnlineSecurity'),
                StreamingTV=request.form.get('StreamingTV'),
                StreamingMovies=request.form.get('StreamingMovies'),
                Contract=request.form.get('Contract'),
                PaperlessBilling=request.form.get('PaperlessBilling'),
                PaymentMethod=request.form.get('PaymentMethod'),
                MonthlyCharges=float(request.form.get('MonthlyCharges')), 
                TotalCharges=float(request.form.get('TotalCharges')), 
                numAdminTickets=int(request.form.get('numAdminTickets')), 
                numTechTickets=int(request.form.get('numTechTickets')) 
            )
            logging.info("CustomData object created from form data.")

            pred_df=data.get_data_as_data_frame()
            logging.info(f"Input DataFrame for prediction:\n{pred_df}")

            logging.info("Initializing PredictPipeline.")
            predict_pipeline=PredictPipeline()
            
            logging.info("Making prediction...")
            results=predict_pipeline.predict(pred_df)
            logging.info(f"Prediction complete. Raw result: {results}")

            churn_status = "Yes (Customer will churn)" if results[0] == 1 else "No (Customer will not churn)"
            
            logging.info(f"Returning prediction result to template: {churn_status}")
            return render_template('index.html', results=churn_status) 
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return render_template('index.html', results=f"Error: {e}")


if __name__=="__main__":
    logging.info("Starting Flask application...")
    app.run(host="0.0.0.0", port=25565, debug=True)