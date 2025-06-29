#To trigger CI

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def split_data(data):
    train, test = train_test_split(
        data,
        test_size=0.4,
        stratify=data["species"],
        random_state=42
    )
    return train, test

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = metrics.accuracy_score(preds, y_test)
    return acc

if __name__ == "__main__":
    data = load_data("data/iris.csv")
    train, test = split_data(data)

    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train["species"]
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test["species"]

    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)

    print(f"The accuracy of the Decision Tree is {acc:.3f}")

