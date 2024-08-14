from utils.model_utils import load_data, preprocess_data, save_data
from sklearn.model_selection import train_test_split

def main(file_path: str):
    """
    Main function to preprocess data.

    Args:
        file_path (str): Path to the CSV file containing the data.
    """
    df = load_data(file_path)
    df, label_encoders, scaler = preprocess_data(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    save_data(X_train, X_test, y_train, y_test, label_encoders, scaler)

if __name__ == "__main__":
    file_path = 'data/raw_data/customer_churn_data_usecase.csv'
    main(file_path)
