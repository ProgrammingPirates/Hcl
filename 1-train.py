from utils.model_utils import load_preprocessed_data, save_model
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def main():
    """
    Main function to train the churn prediction model with SMOTE oversampling.
    """
    # Load preprocessed data
    X_train, _, y_train, _ = load_preprocessed_data()

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize and train the model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict on training data
    y_pred_train = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Training Accuracy: {accuracy}')

    # Save the trained model
    save_model(model, 'model/churn/churn_model.pkl')

if __name__ == "__main__":
    main()
