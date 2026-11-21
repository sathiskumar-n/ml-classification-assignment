"""Model 4: Naive Bayes - Bank Marketing classification."""
from sklearn.naive_bayes import GaussianNB
from data_preprocessing import load_and_preprocess, evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess()

model = GaussianNB()
model.fit(X_train, y_train)

evaluate_model("Naive Bayes", model, X_test, y_test)
