"""Model 3: kNN - Bank Marketing classification."""
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import load_and_preprocess, evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess()

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

evaluate_model("kNN", model, X_test, y_test)
