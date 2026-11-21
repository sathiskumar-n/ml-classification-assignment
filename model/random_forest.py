"""Model 5: Random Forest (Ensemble) - Bank Marketing classification."""
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess, evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess()

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

evaluate_model("Random Forest (Ensemble)", model, X_test, y_test)
