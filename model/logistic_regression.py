"""Model 1: Logistic Regression - Bank Marketing classification."""
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_preprocess, evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

evaluate_model("Logistic Regression", model, X_test, y_test)
