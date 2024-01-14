from sklearn.svm import SVC
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

x_train_tfidf = pickle.load(
    open(os.path.join(BASE_DIR, "./data/x_train_tfidf.pkl"), "rb")
)
y_train = pickle.load(open(os.path.join(BASE_DIR, "./data/y_train.pkl"), "rb"))


model = SVC(C=5.885730113212848, gamma=0.614130186652593, random_state=279).fit(
    x_train_tfidf, y_train
)

os.makedirs(os.path.join(BASE_DIR, "./models"), exist_ok=True)

pickle.dump(model, open(os.path.join(BASE_DIR, "./models/model.pkl"), "wb"))
