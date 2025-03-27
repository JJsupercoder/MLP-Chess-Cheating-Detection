from sklearn.ensemble import ExtraTreesClassifier

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the Extra Trees Classifier model
    model = ExtraTreesClassifier(random_state=42, class_weight='balanced',
                                 max_depth=5, min_samples_split=10000, n_jobs=20)
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