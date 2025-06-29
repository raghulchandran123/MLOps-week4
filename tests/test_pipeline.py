from src import pipeline

def test_load_data():
    data = pipeline.load_data("data/iris.csv")
    assert not data.empty, "Loaded data is empty"

def test_split_data_shape():
    data = pipeline.load_data("data/iris.csv")
    train, test = pipeline.split_data(data)
    assert len(train) > 0
    assert len(test) > 0

def test_model_accuracy():
    data = pipeline.load_data("data/iris.csv")
    train, test = pipeline.split_data(data)
    X_train = train[["sepal_length","sepal_width","petal_length","petal_width"]]
    y_train = train["species"]
    X_test = test[["sepal_length","sepal_width","petal_length","petal_width"]]
    y_test = test["species"]

    model = pipeline.train_model(X_train, y_train)
    acc = pipeline.evaluate_model(model, X_test, y_test)
    assert acc > 0.7, f"Accuracy too low: {acc}"
