import numpy as np

epsilon = np.finfo(np.float64).eps

def diagnose_single_fault(spectra,  # np array [number of tests, number of components] - binary
                          error_vector,  # np array [number of tests] - binary
                          similarity_method, prior=None):

    methods = {  # function(a,b,c,d) -> similarity
        "jaccard": jaccard_similarity,
        "dice": dice_similarity,
        "intersection": intersection_similarity,
        "inner_product": inner_product_similarity,
        "faith": faith_similatiry,
        "cosine": cosine_similarity,
        "prior": prior_only
    }

    a, b, c, d = calc_a_b_c_d(spectra, error_vector)
    similarity_func = methods[similarity_method]
    similarity = similarity_func(a, b, c, d)

    probabilities = similarity.astype(np.float64)
    if prior is not None:
        probabilities *= prior

    # normalize probabilities
    probabilities = probabilities / (probabilities.sum() + epsilon)

    # order diagnoses by probability
    d_order = np.argsort(-probabilities)  # highest prob first
    diagnoses = list(map(int, d_order))
    probabilities = probabilities[d_order]

    return diagnoses, probabilities

def prior_only(a, b, c, d):
    return np.ones(a.shape)

def jaccard_similarity(a, b, c, d):
    return a / ((a + b + c) + epsilon)

def dice_similarity(a, b, c, d):
    return 2 * a / ((2 * a + b + c) + epsilon)

def intersection_similarity(a, b, c, d):
    return a

def inner_product_similarity(a, b, c, d):
    return a + d

def faith_similatiry(a, b, c, d):
    return a + 0.5 * d / ((a + b + c + d) + epsilon)

def cosine_similarity(a, b, c, d):
    epsilon_vec = np.ones(a.shape)*epsilon
    return a / (np.sqrt((a + b) * (a + c)) + epsilon_vec)


def calc_a_b_c_d(spectra, error_vector):
    error_vector = error_vector.reshape(-1, 1)  # column
    failed_filter = error_vector == 1
    pass_filter = np.logical_not(error_vector)
    same_filter = spectra == error_vector
    different_filter = np.logical_not(same_filter)

    a = (same_filter & failed_filter).sum(axis=0)
    b = (different_filter & failed_filter).sum(axis=0)
    c = (different_filter & pass_filter).sum(axis=0)
    d = (same_filter & pass_filter).sum(axis=0)

    return a, b, c, d


if __name__ == '__main__':
    spectra = np.array([[1, 0, 1],
                        [0, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0]])
    error_vector = np.array([1, 1, 1, 1, 0])

    a, b, c, d = calc_a_b_c_d(spectra, error_vector)
    assert (a == np.array([2, 2, 2])).sum() == 3
    assert (b == np.array([2, 2, 2])).sum() == 3
    assert (c == np.array([1, 1, 0])).sum() == 3
    assert (d == np.array([0, 0, 1])).sum() == 3

    jaccard = jaccard_similarity(a, b, c, d)
    assert (jaccard == np.array([0.4, 0.4, 0.5])).sum() == 3

    dice = dice_similarity(a, b, c, d)
    assert (dice == np.array([4/7, 4/7, 4/6])).sum() == 3

    d,p = diagnose_single_fault(spectra, error_vector, "dice")
    assert (d == np.array([2,0,1])).sum() == 3

    spectra = np.array([[1, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0],
                        [0, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1]])
    error_vector = np.array([1, 1, 0, 1, 0])

    sim_methods = ["jaccard","dice", "intersection", "inner_product", "faith", "cosine"]
    for method in sim_methods:
        print(f"method: {method} similarity")
        print(diagnose_single_fault(spectra, error_vector, method))