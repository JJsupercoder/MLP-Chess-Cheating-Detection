from sklearn.naive_bayes import GaussianNB, BernoulliNB  # Using GaussianNB for continuous features

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    model = BernoulliNB()
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