import pandas as pd 
from sklearn.metrics import f1_score, confusion_matrix

train_x = pd.read_csv("data/train_x.csv", index_col = "id")
dev_x = pd.read_csv("data/dev_x.csv",  index_col = "id")
dev_pred = pd.read_csv("data/dev_y_pred_tags_bigram.csv", index_col = "id")
dev_y  = pd.read_csv("data/dev_y.csv",  index_col = "id")
train_x.columns = ["word"]
dev_x.columns = ["word"]
dev_pred.columns = ["predicted"]
dev_y.columns = ["actual"]

data = dev_x.join(dev_y).join(dev_pred)
temp = data[~data["word"].isin(train_x["word"])]
#print(temp)
print("Unknon F1 Score:", f1_score(
    temp.actual,
    temp.predicted,
    average = "weighted"
))
print("Overall F1 Score:", f1_score(
    dev_y.actual,
    dev_pred.predicted,
    average = "weighted"
))