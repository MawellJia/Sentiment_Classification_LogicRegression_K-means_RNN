import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, C):
    """Given a training dataset and a regularisation parameter
    return a logistic regression model fit to the dataset.

    Args:
        X_train: A (sparse or dense) matrix of features of documents.
            Each row is a document represented by its feature vector.
        y_train (np.ndarray): A vector of class labels, each element
            of the vector is either 0 or 1.
        C (float): Regularisation parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # check the input
    assert X_train.shape[0] == y_train.shape[0]
    assert C > 0

    # train the logistic regression classifier
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(X_train, y_train)
    return model


def eval_model(X_test, y_test, model):
    """Given a model already fit to the training data, return the accuracy
        on the provided test data.

    Args:
        model (LogisticRegression): The trained logistic regression model
        X_test: A (sparse or dense) matrix of features of documents.
            Each row is a document represented by its feature vector.
        y_test (np.ndarray): A vector of class labels, each element of the 
            vector is either 0 or 1.

    Returns:
        float: The accuracy of the model on the provided data.
    """
    # check the input
    assert isinstance(model, LogisticRegression)
    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[1] == model.n_features_in_

    # test the logistic regression classifier and calculate the accuracy
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score


def search_C(X_train, y_train, X_val, y_val, return_best_acc=False):
    """Search the best value of hyper-parameter C using the validation set.

    Args:
        X_train, X_val: (Sparse or dense) matrices of document features for
            training and validation, respectively. Each row is a document
            represented by its feature vector.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.
        return_best_acc (boolean): Optional. If True also return the best accuracy
            score on the validation set.

    Returns:
        float: The best C.
        float: Optional. The best accuracy score on the validation set.
    """
    # check the input
    if issparse(X_train):
        assert issparse(X_val)
        assert type(X_train) == type(X_val)
    else:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_train.shape[1] == X_val.shape[1]

    # TODO: search for the best C parameter using the validation set
    print('Searching best hyper parameter (C) value ...')

    # Initialize best_C and best_acc
    best_C = 0
    best_acc = 0

    # Initialize some C_values
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    # C_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # C_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # Find the best one in C_values
    for C in C_values:
        model = train_model(X_train, y_train, C)  # Train the model
        score = eval_model(X_val, y_val, model)  # Get score for different C
        print("C:", C,"acc:", score)
        if score > best_acc:
            best_acc = score
            best_C = C
    print("best C:",best_C)

    return (best_C, best_acc) if return_best_acc else best_C
