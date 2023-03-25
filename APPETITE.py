import copy
import numpy as np

from SFL import get_SFL_for_diagnosis_nodes, get_prior_probs_left_right, get_diagnosis_single_fault
from buildModel import map_tree

def apply_appetite(tree, df, class_name, feature_types, diagnosis_alg="both", nodes_mean_values=None, train_data=None):
    """
    The function apply APPETITE approach on a given decision tree, diagnose the faulty node and return a modified tree according to the new data.
    :param tree: an SKLearn decision tree model. The model will not be modified.
    :param df: pandas dataframe, data after the drift
    :param class_name: the name of the predicted class, as it appears in the dataframe
    :param feature_types: a list of chars [N,B,C] representing the features types in the same order as the features appears in the dataframe.
            N = numeric, B = binary, C = categorical.
    :param diagnosis_alg: text parameter for the diagnosis algorithm, "sfl" for SFL-DS, "stat" for STAT-AN and "both" for the combination of them
    :param nodes_mean_values: a list of node's feature mean on trained data.
    :param train_data: the data used to train the model, pandas dataframe
            should not be None in case nodes_mean_values is not provided
    :return: the diagnosis of the faulty nodes & a fixed tree according to the diagnosis
    """
    assert nodes_mean_values is not None or train_data is not None

    model_rep = map_tree(tree)

    data_y = df[class_name]
    features = df.columns.remove(class_name)
    data_x = df[features]
    prediction = tree.predict(data_x)
    samples = data_x, prediction, data_y

    # diagnose faulty nodes
    diagnosis = diagnose_single_node(tree, samples, model_rep, diagnosis_alg)

    # fix the tree
    fixed_model = fix_model(tree, model_rep, diagnosis, samples, feature_types, nodes_mean_values, train_data)

    return diagnosis, fixed_model


def diagnose_single_node(orig_model, new_data, model_rep, diagnosis_alg):
    similarity_method = {"sfl":"faith",
                         "stat": "prior",
                         "both": "faith"
                         }

    _, spectra, error_vector, _ = get_SFL_for_diagnosis_nodes(orig_model, new_data, model_rep)
    priors = get_prior_probs_left_right(model_rep, spectra)
    if diagnosis_alg == "sfl":
        priors = None

    method = similarity_method[diagnosis_alg]
    diagnoses, probabilities = get_diagnosis_single_fault(spectra, error_vector, method, priors=priors)

    return diagnoses[0]

def fix_model(orig_model, model_rep, node, samples, feature_types, nodes_mean_values, train_data):
    to_fix = copy.deepcopy(orig_model)
    data_x, prediction, data_y = samples

    # check if the faulty node is a leaf
    if model_rep[node]["left"] == -1:
        return fix_leaf(to_fix, node, data_y)

    feature = orig_model.tree_.feature[node]
    f_type = feature_types[feature]

    if f_type in ("B", "C"):
        return fix_categorical(to_fix, node)
    else: # f_type == N
        if nodes_mean_values is not None:
            node_old_mean = nodes_mean_values[node]
        else:
            # compute node's mean from train data
            node_old_data = filter_data_for_node(model_rep, node, train_data)
            feature_name = data_x.columns[int(feature)]
            node_old_mean = node_old_data[feature_name].mean()

        return fix_numeric(to_fix, model_rep, node, data_x, node_old_mean)


def fix_leaf(tree, node, labels):
    common_class = labels.value_counts().idxmax()

    # modify the tree in such way that SKLearn will classify common_class
    values = tree.tree_.value[node]
    max_count = np.max(values) + 1
    values[0][common_class] = max_count
    tree.tree_.value[node] = values

    return tree

def fix_numeric(tree, tree_rep, node, data_x, node_old_mean):
    # calculate node's diff
    feature = tree.tree_.feature[node]
    feature_name = data_x.columns[int(feature)]
    node_data = filter_data_for_node(tree_rep, node, data_x)
    node_new_mean = node_data[feature_name].mean()
    diff = node_new_mean - node_old_mean

    # modify tree
    new_threshold = tree.tree_.threshold[node] + diff
    tree.tree_.threshold[node] = new_threshold
    return tree

def fix_categorical(tree, node):
    left_child = tree.tree_.children_left[node]
    right_child = tree.tree_.children_right[node]
    tree.tree_.children_left[node] = right_child
    tree.tree_.children_right[node] = left_child
    return tree

def filter_data_for_node(tree_rep, node, data_x):
    filtered_data = data_x.copy()
    filtered_data["true"] = 1
    indexes_filtered_data = (filtered_data["true"] == 1) # all true
    filtered_data = filtered_data.drop(columns=["true"])

    conditions = tree_rep[node]["condition"]
    for cond in conditions:
        feature = cond["feature"]
        sign = cond["sign"]
        thresh = cond["thresh"]
        feature_name = data_x.columns[int(feature)]
        if sign == ">":
            indexes_filtered = filtered_data[feature_name] > thresh
        else:  # <=
            indexes_filtered = filtered_data[feature_name] <= thresh
        indexes_filtered_data = indexes_filtered & indexes_filtered_data

    return filtered_data[indexes_filtered_data]

if __name__ == '__main__':
    pass
