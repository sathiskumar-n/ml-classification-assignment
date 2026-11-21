"""Model 2: Decision Tree - Bank Marketing classification."""
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import load_and_preprocess, evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess()

model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

evaluate_model("Decision Tree", model, X_test, y_test)
