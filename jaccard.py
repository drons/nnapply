from keras import backend as K

def jaccard_index(y_true, y_pred):
    """
    see https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance
    """
    eps = 1e-7
    intersection = K.sum( K.minimum(y_true, y_pred))
    union = K.sum(K.maximum(y_true, y_pred))
    return (intersection + eps) / (union + eps)


def jaccard_distance(y_true, y_pred):
    """
    see https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance
    """
    return 1.0 - jaccard_index(y_true, y_pred)