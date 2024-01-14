import pickle
from sklearn.metrics import accuracy_score, f1_score

model = pickle.load(open("./models/model.pkl", "rb"))

x_test_tfidf = pickle.load(open("./data/x_test_tfidf.pkl", "rb"))
y_test = pickle.load(open("./data/y_test.pkl", "rb"))

y_pred = model.predict(x_test_tfidf)
print(
    f"\tAccuracy: {accuracy_score(y_test, y_pred)}\n",
    f"\tF1-score: {f1_score(y_test, y_pred)}\n",
)
