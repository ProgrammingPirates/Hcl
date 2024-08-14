from utils.model_utils import load_preprocessed_data, load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    """
    Main function to test the churn prediction model.
    """
    _, X_test, _, y_test = load_preprocessed_data()
    model = load_model('model/churn/churn_model.pkl')
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)
    
    print(f'Test Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

if __name__ == "__main__":
    main()
