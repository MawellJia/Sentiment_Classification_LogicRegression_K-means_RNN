import numpy as np
from gensim.models import Word2Vec
from features import get_features_w2v
from classifier import search_C


def search_hyperparams(Xt_train, y_train, Xt_val, y_val):
    """Search the best values of hyper-parameters for Word2Vec as well as the
    regularisation parameter C for logistic regression, using the validation set.

    Args:
        Xt_train, Xt_val (list(list(list(str)))): Lists of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens) for training and validation, respectively.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.

    Returns:
        dict(str : union(int, float)): The best values of hyper-parameters.
    """
    # TODO: tune at least two of the many hyper-parameters of Word2Vec 
    #       (e.g. vector_size, window, negative, alpha, epochs, etc.) as well as
    #       the regularisation parameter C for logistic regression
    #       using the validation set.
    best_params = dict()

    # Initialize the best accuracy
    best_acc = 0

    # Define two hyper-parameters that we want to tune
    vector_size = [100, 200, 300]
    windows = [5, 10, 20]

    # Change the relative parameters
    for vector in vector_size:
        for window in windows:
            # Train a model in different parameters
            model = train_w2v(Xt_train, vector_size=vector, window=window)

            # Transfer tokenized sentences to vectors
            X_train_vector = get_features_w2v(Xt_train, model)
            X_val_vector = get_features_w2v(Xt_val, model)

            # Get relative C and acc for different combinations
            C, acc = search_C(X_train_vector, y_train, X_val_vector, y_val, return_best_acc=True)

            print('vector:', vector, 'window:', window, 'C:', C, 'accuracy:', acc)

            # Find the best accuracy
            if acc > best_acc:
                best_acc = acc
                best_params['C'] = C
                best_params['vector_size'] = vector
                best_params['window'] = window
    print('best_params:', best_params)
    assert 'C' in best_params
    return best_params


def train_w2v(Xt_train, vector_size=200, window=5, min_count=5, negative=10, epochs=5, seed=101, workers=1,
              compute_loss=False, **kwargs):
    """Train a Word2Vec model.

    Args:
        Xt_train (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        for descriptions of the other arguments.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: A mapping from words (string) to their embeddings
            (np.ndarray)
    """
    sentences_train = [sent for doc in Xt_train for sent in doc]

    # TODO: train the Word2Vec model
    print(f'Train word2vec using {len(sentences_train):,d} sentences ...')

    # Train a Word2Vec model
    w2v_model = Word2Vec(sentences=sentences_train, vector_size=vector_size, window=window, min_count=min_count,
                         negative=negative, epochs=epochs, seed=seed, workers=workers, compute_loss=compute_loss,
                         **kwargs)

    return w2v_model.wv  # Return only word vectors
