from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from utils.llm_utils import convert_natural_language_to_json, GROQ_API_KEY
from langchain_groq import ChatGroq
from utils.model_utils import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(max_retries=1, model='llama3-70b-8192', api_key='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', temperature=0, max_tokens=2048)

# Load the trained model, label encoders, and scaler
try:
    model = load_model('model/churn/churn_model.pkl')
    label_encoders = load_model('model/label_encoder/label_encoders.pkl')
    scaler = load_model('model/scaler/scaler.pkl')
    logger.info("Model, label encoders, and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model, label encoders, or scaler: {e}")
    raise

app = FastAPI()

class CustomerDescription(BaseModel):
    customer_description: str

def get_pred(input_dict):
    """
    This function processes input data, encodes categorical variables, normalizes numerical data,
    and returns the churn prediction along with the confidence score.

    Args:
        input_dict (dict): Input data in dictionary format.

    Returns:
        dict: Dictionary containing the churn prediction and confidence score.
    """
    data = pd.DataFrame([input_dict])
    logger.info(f"Input data: {data}")

    # Encode categorical variables using the label encoders
    for column, le in label_encoders.items():
        if column in data.columns:
            data[column] = le.transform(data[column])
    logger.info(f"Encoded data: {data}")

    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if 'Churn' in numerical_columns:
        numerical_columns = numerical_columns.drop('Churn')
    data[numerical_columns] = scaler.transform(data[numerical_columns])
    logger.info(f"Normalized data: {data}")

    # Make prediction and get confidence score
    try:
        prediction = model.predict(data)[0]
        confidence = model.predict_proba(data)[0][1]  # Probability of the positive class (churn)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise

    logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
    return {'Churn': bool(prediction), 'Confidence': confidence}

@app.post("/predict-from-text")
def predict_from_text(customer_description: CustomerDescription):
    """
    This endpoint accepts a customer description in natural language, converts it into structured
    JSON format using Groq LLM, and predicts customer churn using the pre-trained model.

    Args:
        customer_description (CustomerDescription): A pydantic model containing the customer description.

    Returns:
        str: A message indicating whether the customer is predicted to churn or not, along with the confidence score.
    """
    try:
        natural_language_input = customer_description.customer_description

        # Handle basic greetings and common questions
        if not natural_language_input:
            return 'Customer Description is required!'
        
        if str(natural_language_input).lower().replace('!','') in ('hi', 'hello'):
            return 'Hi, How can I help you!'
        
        if str(natural_language_input).lower().replace('!','').replace('?','') == 'who are you':
            return 'A chatbot that predicts the customer churn based on Natural Language!'
        
        if len(natural_language_input.split()) < 10 or len(natural_language_input) < 20:
            return "Please provide more details about the customer to help you!"

        # Convert natural language input to JSON using Groq LLM
        output = convert_natural_language_to_json(llm=llm, query=natural_language_input)
        logger.info(f"Converted natural language to JSON: {output}")

        # Get prediction and confidence
        prediction = get_pred(input_dict=output)
        churn_prediction = prediction['Churn']
        confidence_score = round(prediction['Confidence'] * 100, 3)
        logger.info(f"Confidence score: {confidence_score}")

        if churn_prediction:
            return f'The customer is predicted to churn. Consider taking actions to retain the customer. Confidence: {confidence_score}%'
        return f'The customer is not predicted to churn. Keep up the good work in retaining the customer. Confidence: {confidence_score}%'

    except Exception as e:
        logger.error(f"Error in processing request: {e}")
        raise HTTPException(status_code=400, detail=f"Error in processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
