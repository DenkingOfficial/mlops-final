import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


tqdm.pandas()
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")

df = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"))

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    re.UNICODE,
)
url_pattern = re.compile(
    r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
)
mail_pattern = re.compile(r"^([a-z0-9_\.-]+)@([a-z0-9_\.-]+)\.([a-z\.]{2,6})$")
html_pattern = re.compile(r"<.*?>")
punctuation_pattern = re.compile(r"[^\w\s\d]+")
braces_pattern = re.compile(r"\([^)]*\)")
joined_pattern = re.compile(r"([a-zа-я])([A-ZА-Я])")
digits_pattern = re.compile(r"\d")

stop_words = set(stopwords.words("english"))


def clean_comment(comment):
    temp_comment = re.sub(braces_pattern, r"", comment)
    temp_comment = re.sub(emoji_pattern, r"", temp_comment)
    temp_comment = re.sub(url_pattern, r"", temp_comment)
    temp_comment = re.sub(mail_pattern, r"", temp_comment)
    temp_comment = re.sub(html_pattern, r"", temp_comment)
    temp_comment = re.sub(punctuation_pattern, r"", temp_comment)
    temp_comment = re.sub(joined_pattern, r"\1 \2", temp_comment)
    temp_comment = re.sub(digits_pattern, "#", temp_comment)
    temp_comment = remove_suffix(temp_comment, " ")
    temp_comment = temp_comment.replace("\xad", "")
    temp_comment = temp_comment.replace("\r", ". ")
    temp_comment = temp_comment.replace("\n", ". ")
    return temp_comment


def tokenize(text):
    text = text.lower()
    text = word_tokenize(text, language="english")
    text = [word for word in text if word.isalpha() and word not in stop_words]
    return text


def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def join_tokens(tokens):
    return " ".join(tokens)


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


df["text"] = df["text"].progress_apply(clean_comment)
df.dropna(inplace=True)
df["tokens"] = df["text"].progress_apply(tokenize).progress_apply(lemmatize)
df["processed"] = df["tokens"].progress_apply(join_tokens)

x_train, x_test, y_train, y_test = train_test_split(
    df["processed"],
    df["target"],
    test_size=0.25,
    random_state=1905,
)

tfidf = TfidfVectorizer(
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)


with open(os.path.join(BASE_DIR, "data/x_train_tfidf.pkl", "wb")) as f:
    pickle.dump(x_train_tfidf, f)

with open(os.path.join(BASE_DIR, "data/x_test_tfidf.pkl", "wb")) as f:
    pickle.dump(x_test_tfidf, f)

with open(os.path.join(BASE_DIR, "data/y_train.pkl", "wb")) as f:
    pickle.dump(y_train, f)

with open(os.path.join(BASE_DIR, "data/y_test.pkl", "wb")) as f:
    pickle.dump(y_test, f)
