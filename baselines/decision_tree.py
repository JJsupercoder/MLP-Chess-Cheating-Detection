from sklearn.tree import DecisionTreeClassifier

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the Decision Tree model
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')  # Added random_state for reproducibility
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