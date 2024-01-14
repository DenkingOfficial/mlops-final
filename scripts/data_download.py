from dvc.api import DVCFileSystem
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fs = DVCFileSystem(BASE_DIR)
fs.get_file("data/train.csv", os.path.join(BASE_DIR, "data/train.csv"))
