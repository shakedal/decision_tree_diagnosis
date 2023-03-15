from sklearn import metrics
from sklearn.tree import _tree, export_text
# from sklearn.tree.export import export_text
from DataSet import DataSet
from DecisionTree import DecisionTree
from SFL import PARENTS
from buildModel import build_model, map_tree
import numpy as np
import copy

def change_leaf_classification(tree, nodes, leaf_fix_type="second", dataset=None, tree_rep=None):
    if leaf_fix_type == "second":
        # switch between first classification to second classification (second most popular in node)
        for node in nodes:
            values = tree.tree_.value[node]
            val_order = np.argsort(-values)[0]  # highest first
            first = val_order[0]
            first_val = values[0][first]
            second = val_order[1]
            values[0][first] = values[0][second]
            values[0][second] = first_val
            tree.tree_.value[node] = values

    elif leaf_fix_type == "new_data":
        # classification is most common class in node after drift
        assert (dataset is not None)
        assert (tree_rep is not None)
        for node in nodes:
            filtered_data = filter_data_for_node(tree_rep, node, dataset, "after")
            labels = filtered_data[dataset.target]
            if labels.size < 1:
                continue
            common_class = labels.value_counts().idxmax()

            values = tree.tree_.value[node]
            max_count = np.max(values) +1
            values[0][common_class] = max_count
            tree.tree_.value[node] = values

    else:
        assert 1 < 0, f"leaf fix type illegal: {leaf_fix_type}"

    return tree

def change_tree_threshold(tree, nodes, thresholds):
    for i in range(len(nodes)):
        tree.tree_.threshold[nodes[i]] = thresholds[i]
    return tree

def change_tree_selection(tree, nodes):
    for node in nodes:
        left_child = tree.tree_.children_left[node]
        right_child = tree.tree_.children_right[node]
        tree.tree_.children_left[node] = right_child
        tree.tree_.children_right[node] = left_child
        print("node id: {}".format(node))
        print("left child: {}".format(left_child))
        print("right child: {}".format(right_child))
    return tree

def find_nodes_threshold_from_diagnosis(model, diagnosis_features, features_diff):
    diagnosis_features = list(int(x / 2) for x in diagnosis_features)

    nodes = list()
    thresholds = list()

    node_count = model.tree_.node_count
    for i in range(node_count):
        feature = model.tree_.feature[i]
        if feature in diagnosis_features:
            nodes.append(i)
            new_threshold = model.tree_.threshold[i] + features_diff[feature]
            thresholds.append(new_threshold)
            # print("update node {} with feature {}, change threshold by {}".format(i, feature, new_threshold))

    return nodes, thresholds

def print_tree_rules(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print( "def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print( "{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print( "{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print( "{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

def change_nodes_threshold(model, nodes, features_diff):
    for node in nodes:
        feature = model.tree_.feature[node]
        if feature == -2: # node is a leaf - no threshold
            continue
        # print(f"node {node} feature is: {feature}")
        new_threshold = model.tree_.threshold[node] + features_diff[feature]
        model.tree_.threshold[node] = new_threshold
    return model

def change_nodes_by_type(model, nodes,feature_types, features_diff,
                         diff_type="all", leaf_fix_type="new_data", tree_rep=None, dataset=None):
    binary_categorical = list()
    numeric = list()
    leaves = list()
    for node in nodes:
        feature = model.tree_.feature[node]
        if feature == -2:  # node is a leaf
            leaves.append(node)
        f_type = feature_types[feature]
        if f_type == "numeric":
            numeric.append(node)
        elif f_type == "binary" or f_type == "categorical":
            binary_categorical.append(node)

    if diff_type == "all":
        model = change_nodes_threshold(model, numeric, features_diff)
    else:
        assert (dataset is not None)
        model = change_nodes_threshold_only_node(model, tree_rep, numeric, dataset)
    model = change_tree_selection(model, binary_categorical)
    model = change_leaf_classification(model, leaves, leaf_fix_type, dataset, tree_rep)
    return model

def get_parents(nodes):
    parents = set()
    for node in nodes:
        parent = PARENTS[node]
        parents.add(parent)
    return parents

def train_subtree(model, node, dataset, tree_rep):
    if node == 0:
        data = dataset.data.iloc[dataset.before_size: dataset.before_size + dataset.after_size].copy()
        new_model = build_model(data, dataset.features, dataset.target, to_split=True)
        return new_model

    filtered_data = filter_data_for_node(tree_rep, node, dataset, "after")
    if len(filtered_data) == 0:
        return -1
    new_subtree = build_model(filtered_data, dataset.features, dataset.target, to_split=True)

    print("TREE:")
    print_tree_rules(new_subtree, dataset.features)
    print(f"new tree size: {new_subtree.tree_.node_count}")

    fixed_model = DecisionTree(model)
    fixed_model.replace_subtree(node, new_subtree)
    return fixed_model

def filter_data_for_node(tree_rep, node, dataset, data_type):
    if data_type == "before":
        filtered_data = dataset.data.iloc[:dataset.before_size].copy()
    elif data_type == "after":
        filtered_data = dataset.data.iloc[dataset.before_size: dataset.before_size + dataset.after_size].copy()
    else:  # test set
        filtered_data = dataset.data.iloc[len(dataset.data) - dataset.test_size:-1].copy()

    filtered_data["true"] = 1
    indexes_filtered_data = (filtered_data["true"] == 1) # all true
    filtered_data = filtered_data.drop(columns=["true"])

    conditions = tree_rep[node]["condition"]
    for cond in conditions:
        feature = cond["feature"]
        sign = cond["sign"]
        thresh = cond["thresh"]
        feature_name = dataset.features[int(feature)]
        if sign == ">":
            indexes_filtered = filtered_data[feature_name] > thresh
        else:  # <=
            indexes_filtered = filtered_data[feature_name] <= thresh
        indexes_filtered_data = indexes_filtered & indexes_filtered_data

    return filtered_data[indexes_filtered_data]

def change_nodes_threshold_only_node(model, tree_rep, diagnosis, dataset):
    for node in diagnosis:
        feature = model.tree_.feature[node]
        if feature == -2:
            continue
        feature_name = dataset.features[feature]

        # filter data for node
        data_before = filter_data_for_node(tree_rep, node, dataset, "before")
        data_after = filter_data_for_node(tree_rep, node, dataset, "after")

        # calculate diff
        mean_before = data_before[feature_name].mean()
        mean_after = data_after[feature_name].mean()
        diff = mean_after - mean_before

        # change threshold
        new_threshold = model.tree_.threshold[node] + diff
        model.tree_.threshold[node] = new_threshold

    return model


if __name__ == '__main__':
    pass


