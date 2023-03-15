import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from DataSet import DataSet

param_grid_tree = {
    "criterion": ["gini", "entropy"],
    # "min_samples_split": [2, 0.1, 0.05, 3],
    # "max_depth": [4, 6, 8, 10, 12],
    # "min_samples_leaf": [1, 2, 5, 10],
    "max_leaf_nodes": [10, 20, 30]
}


def build_model(data, features, target, model_type="tree", to_split=False, val_data=None):
    np.random.seed(0)

    x_train_all = data[features]
    y_train_all = data[target]

    if to_split:  # split to train and validation sets
        if len(x_train_all) > 1:
            x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=7)  # 80% training and 20% test
        else:
            x_train, x_val = x_train_all, x_train_all
            y_train, y_val = y_train_all, y_train_all
        # print(f"x train:\n{x_train}")
        # print(f"x val:\n{x_val}")
    else:  # do not split, use validation data given
        assert val_data is not None
        x_val = val_data[features]
        y_val = val_data[target]
        x_train, y_train = x_train_all, y_train_all

    x_train1 = x_train
    y_train1 = y_train
    all_y = y_train.unique()
    y_count = y_train.value_counts()
    min_y_count = y_count.min()
    if min_y_count == 1:
        only_1 = np.where(y_count == 1)
        for y_loc in only_1:  # add another sample to each class
            sample_filter = y_train == all_y[y_loc[0]]
            x_train1 = x_train1.append(x_train.loc[sample_filter,features], ignore_index=True)
            y_train1 = y_train1.append(y_train[sample_filter], ignore_index=True)
        min_y_count = 2
    n_split = min(5, min_y_count)

    # choose best parameters
    dec_tree = DecisionTreeClassifier()
    clf_GS = GridSearchCV(estimator=dec_tree, param_grid=param_grid_tree, cv=n_split)
    clf_GS.fit(x_train1, y_train1)
    best_params = clf_GS.best_params_
    # print(best_params)

    # train tree on best params and then prune
    clf = DecisionTreeClassifier(criterion=best_params["criterion"], #max_depth=best_params["max_depth"],
                                     max_leaf_nodes=best_params["max_leaf_nodes"])
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = set(path.ccp_alphas)
    clfs = []
    accuracy_val = []
    for ccp_alpha in ccp_alphas:
        if ccp_alpha < 0:
            continue
        clf = DecisionTreeClassifier(criterion=best_params["criterion"], #max_depth=best_params["max_depth"],
                                     max_leaf_nodes=best_params["max_leaf_nodes"], ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        pred = clf.predict(x_val)
        accuracy = metrics.accuracy_score(y_val, pred)
        accuracy_val.append(accuracy)

    accuracy_val = np.array(accuracy_val)
    best_accuracy = np.max(accuracy_val)
    # print(best_accuracy)
    best = np.where(accuracy_val == best_accuracy)[0][-1]  # best val accuracy, max pruning
    # print(best)
    best_clf = clfs[best]
    return best_clf

def map_tree(tree):
    class_names = tree.classes_
    tree_representation = {"classes": class_names,
                           0: {"depth": 0,
                               "parent": -1}}
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)

        left_child = tree.tree_.children_left[node]
        tree_representation[node]["left"] = left_child
        if left_child != -1:
            tree_representation[left_child] = {"parent":node,
                                               "type": "left"}
            nodes_to_check.append(left_child)
        right_child = tree.tree_.children_right[node]
        tree_representation[node]["right"] = right_child
        if right_child != -1:
            tree_representation[right_child] = {"parent":node,
                                               "type": "right"}
            nodes_to_check.append(right_child)

        tree_representation[node]["feature"] = tree.tree_.feature[node]
        tree_representation[node]["threshold"] = tree.tree_.threshold[node]

        tree_representation[node]["value"] = tree.tree_.value[node]

        if node != 0:
            parent = tree_representation[node]["parent"]
            tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
            parent_cond = tree_representation[parent]["condition"]
            sign = "<=" if tree_representation[node]["type"] == "left" else ">"
            #cond = f"{model.tree_.feature[parent]} {sign} {model.tree_.threshold[parent]}"
            cond = {
                "feature": tree.tree_.feature[parent],
                "sign": sign,
                "thresh": tree.tree_.threshold[parent]
            }
            tree_representation[node]["condition"] = parent_cond + [cond]
        else:  # root
            tree_representation[node]["condition"] = []

        if left_child == -1:  # leaf
            value = tree.tree_.value[node]
            class_name = np.argmax(value)
            class_name = class_names[class_name]
            tree_representation[node]["class"] = class_name

    node_list = list(tree_representation.keys())
    node_list.remove("classes")
    leaf_nodes = list(filter(lambda n: tree_representation[n]["left"] == -1, node_list))
    nodes_to_check = leaf_nodes
    # nodes_to_check = sorted(leaf_nodes, reverse=True)
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(nodes_to_check.index(max(nodes_to_check)))
        if node == -1:
            break
        # node = nodes_to_check.pop(0)
        if tree_representation[node].get("subtree") is not None:
            continue
        left_child = tree_representation[node]["left"]
        right_child = tree_representation[node]["right"]
        subtree = [node]
        if left_child != -1:
            subtree += tree_representation[left_child]["subtree"]
        if right_child != -1:
            subtree += tree_representation[right_child]["subtree"]
        tree_representation[node]["subtree"] = subtree

        nodes_to_check.append(tree_representation[node]["parent"])

    return tree_representation

def prune_tree(tree, tree_rep):
    node_list = list(range(tree.tree_.node_count))
    leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
    pruned = []
    while len(leaf_nodes) > 0:
        leaf = leaf_nodes.pop()
        if leaf == 0:
            break
        parent = tree_rep[leaf]["parent"]
        class_name = tree_rep[leaf]["class"]
        if tree_rep[leaf]["type"] == "left":
            brother = tree_rep[parent]["right"]
        else:
            brother = tree_rep[parent]["left"]

        if brother in leaf_nodes and tree_rep[brother]["class"] == class_name:  # prune
            leaf_nodes.remove(brother)
            pruned += [leaf, brother]

            value_leaf = tree.tree_.value[leaf]
            value_brother = tree.tree_.value[brother]
            value_parent = tree.tree_.value[parent]
            assert (value_parent == value_brother+value_leaf).all(), f"values parent: {value_parent} is not equal to brother values\nvalue leaf: {value_leaf} + value brother {value_brother}"

            tree.tree_.children_left[parent] = -1
            tree.tree_.children_right[parent] = -1
            tree.tree_.feature[parent] = -2
            tree_rep[parent]["class"] = class_name

            leaf_nodes.append(parent)

    print(f"pruned {len(pruned)} nodes, list: {pruned}")
    return tree

def calculate_error(tree_rep):
    error_dictionary = {}

    def calculate_node_error(node, tree_rep):
        left = tree_rep[node]["left"]
        right = tree_rep[node]["right"]

        if left == -1: #leaf
            values = tree_rep[node]["value"].copy()
            n_samples = values.sum()
            class_num = values.argmax()
            error_rate = 1 - (values[0][class_num]/n_samples)
            error_dictionary[node] = error_rate
            return error_rate

        else: # non-terminal node
            n_samples = tree_rep[node]["value"].sum()

            if left in error_dictionary:
                left_error = error_dictionary[left]
            else:
                left_error = calculate_node_error(left, tree_rep)
            n_samples_left = tree_rep[left]["value"].sum()

            if right in error_dictionary:
                right_error = error_dictionary[right]
            else:
                right_error = calculate_node_error(right, tree_rep)
            n_samples_right = tree_rep[right]["value"].sum()

            assert n_samples == n_samples_right + n_samples_left
            error_rate = (right_error*n_samples_right + left_error*n_samples_left)/n_samples
            error_dictionary[node] = error_rate
            return error_rate

    calculate_node_error(0, tree_rep)
    return error_dictionary

def calculate_left_right_ratio(tree_rep):
    left_right_dict = {}
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)
        left = tree_rep[node]["left"]
        right = tree_rep[node]["right"]

        if left == -1:  # leaf
            left_right_dict[node] = -1
        else:
            went_left = tree_rep[left]["value"].sum()
            went_right = tree_rep[right]["value"].sum()
            total = tree_rep[node]["value"].sum()
            assert went_left + went_right == total

            left_right_dict[node] = went_left / total
            nodes_to_check.append(left)
            nodes_to_check.append(right)
    return left_right_dict

def print_tree_rules(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)


if __name__ == '__main__':
    pass