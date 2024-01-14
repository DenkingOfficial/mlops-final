from sklearn.svm import SVC
import pickle
import os

x_train_tfidf = pickle.load(open("./data/x_train_tfidf.pkl", "rb"))
y_train = pickle.load(open("./data/y_train.pkl", "rb"))


model = SVC(C=5.885730113212848, gamma=0.614130186652593, random_state=279).fit(
    x_train_tfidf, y_train
)

os.makedirs("./models", exist_ok=True)

pickle.dump(model, open("./models/model.pkl", "wb"))
