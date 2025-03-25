from sklearn.linear_model import LogisticRegression

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')  # Increase max_iter if needed
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")

if __name__ == '__main__':
    train()