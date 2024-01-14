from dvc.api import DVCFileSystem

fs = DVCFileSystem("./")
fs.get_file("data/train.csv", "data/train.csv")
