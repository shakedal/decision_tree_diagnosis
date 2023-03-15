# from sfl.Diagnoser.diagnoserUtils import write_json_planning_file, readPlanningFile
import numpy as np
# from Barinel import calculate_diagnoses_and_probabilities_barinel_shaked
from SingleFault import diagnose_single_fault
from buildModel import calculate_error, calculate_left_right_ratio

THRESHOLD = 0.1
ONLY_POSITIVE = True
MATRIX_FILE_PATH = 'matrix_for_SFL1'
PARENTS = dict()
epsilon = np.finfo(np.float64).eps

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def get_matrix_entrance(sample_id, samples, node_indicator, conflicts):
    data_x, prediction, labels = samples
    node_index = node_indicator.indices[            # extract the relevant path for sample_id
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ].tolist()
    result = 0 if prediction[sample_id] == labels.values[sample_id] else 1  # 1 if there is an error in prediction
    test_detail = f"T{sample_id};{node_index};{result}"

    if result == 1: # classification is wrong
        conflicts.add(tuple(node_index))

    return test_detail, conflicts

def get_SFL_for_diagnosis_nodes(model, samples, model_rep):
    BAD_SAMPLES = list()
    data_x, prediction, labels = samples
    number_of_samples = len(data_x)
    number_of_nodes = model.tree_.node_count

    # initialize spectra and error vector
    error_vector = np.zeros(number_of_samples)
    spectra = np.zeros((number_of_samples, number_of_nodes))

    node_indicator = model.decision_path(data_x)  # get paths for all samples
    conflicts = set()
    errors = 0
    for sample_id in range(number_of_samples):
        node_index = node_indicator.indices[  # extract the relevant path for sample_id
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ].tolist()
        for node_id in node_index:
            # set as a component in test
            spectra[sample_id][node_id] = 1
        if prediction[sample_id] != labels.values[sample_id]:  # test result is "fail"
            error_vector[sample_id] = 1
            errors += 1
            conflicts.add(tuple(node_index))
            BAD_SAMPLES.append(sample_id)

    print(f"Conflicts: {conflicts}")
    print(f"Number of misclassified samples: {errors}")
    return BAD_SAMPLES, spectra, error_vector, conflicts

def get_prior_probs_depth(model_rep, number_of_nodes):
    # define prior vector
    priors = np.ones(number_of_nodes) * 0.99
    depth = [model_rep[node]["depth"] if node in model_rep else 0 for node in range(number_of_nodes)]
    max_depth = max(depth)
    # priors = [
    #     0.99**(max_depth - depth[node])
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.99**((max_depth - depth[node])*4)
    #     for node in range(number_of_nodes)]
    # priors = [  # BEST FOR: barinel single node
    #     0.01 * (depth[node]+1)
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.01 * (depth[node]+1)/4
    #     for node in range(number_of_nodes)]
    # priors = [
    #     0.1 / (max_depth - depth[node] + 1)
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.1 / (4*(max_depth - depth[node] + 1))
    #     for node in range(number_of_nodes)]
    priors = [  # BEST FOR: barinel original
        1 - ((max_depth - depth[node] + 1) / (max_depth + 2))
        if node in model_rep and model_rep[node]["left"] != -1
        else (1 - ((max_depth - depth[node] + 1) / (max_depth + 2))) / 4
        for node in range(number_of_nodes)]
    priors = np.array(priors)
    return priors

def get_prior_probs_left_right(model_rep, spectra):
    # define prior vector
    number_of_nodes = spectra.shape[1]
    priors = calculate_left_right_diff(spectra, model_rep)
    # priors = softmax(priors)
    priors = np.array(priors)
    return priors

# def get_diagnosis_barinel(spectra, error_vector, priors):
#     diagnoses, probabilities = calculate_diagnoses_and_probabilities_barinel_shaked(spectra, error_vector, priors)
#     return diagnoses, probabilities

def get_diagnosis_single_fault(spectra, error_vector, similarity_method,priors=None):
    diagnoses, probabilities = diagnose_single_fault(spectra, error_vector, similarity_method, priors)
    return diagnoses, probabilities

def calculate_nodes_error(spectra, error_vector):
    participation = spectra.sum(axis=0)
    errors = (spectra * (error_vector.reshape(-1,1))).sum(axis=0)
    error_rate = errors / (participation + epsilon)
    return error_rate

def calculate_left_right_diff(spectra, model_rep):
    n_nodes = spectra.shape[1]
    left_right_dict = calculate_left_right_ratio(model_rep)
    original_ratio = np.zeros(n_nodes)
    for node, ratio in left_right_dict.items():
        original_ratio[node] = ratio

    participation = spectra.sum(axis=0)
    left_right_current = np.zeros(n_nodes)
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)
        left = model_rep[node]["left"]
        right = model_rep[node]["right"]

        if left != -1:  # not a leaf
            total = participation[node] + epsilon
            went_left = participation[left]
            left_right_current[node] = went_left / total
            nodes_to_check.append(left)
            nodes_to_check.append(right)
        else:
            left_right_current[node] = -1

    diff_ratio = np.absolute(left_right_current - original_ratio)
    return diff_ratio

def get_diagnosis_error_rate(spectra, error_vector, model_rep):
    n_nodes = spectra.shape[1]
    original_errors_dict = calculate_error(model_rep)
    original_errors = np.zeros(n_nodes)
    for node, error_rate in original_errors_dict.items():
        original_errors[node] = error_rate

    current_errors = calculate_nodes_error(spectra, error_vector)
    for node in range(n_nodes):
        cur_error_rate = current_errors[node]
        if np.isnan(cur_error_rate):
            current_errors[node] = 0

    diff_error = current_errors - original_errors
    #diff_error = (current_errors - original_errors) / (original_errors + epsilon)
    # diff_error = (current_errors - original_errors) * (np.power(original_errors + epsilon, 0.9)/(original_errors + epsilon))
    d_order = np.argsort(-diff_error)
    diagnoses = list(map(int, d_order))
    rank = diff_error[d_order]
    return diagnoses, rank

def get_diagnosis_left_right(spectra, error_vector, model_rep):
    diff_ratio = calculate_left_right_diff(spectra, model_rep)
    d_order = np.argsort(-diff_ratio)
    diagnoses = list(map(int, d_order))
    rank = diff_ratio[d_order]
    return diagnoses, rank

