import numpy as np

from DataSet import DataSet
from buildModel import build_model


class DecisionTree:
    def __init__(self, sklearn_tree, tree_rep=None):
        self.node_count = sklearn_tree.tree_.node_count
        self.max_depth = sklearn_tree.tree_.max_depth

        self.threshold = sklearn_tree.tree_.threshold
        self.feature = sklearn_tree.tree_.feature
        self.children_left = sklearn_tree.tree_.children_left
        self.children_right = sklearn_tree.tree_.children_right

        self.value = sklearn_tree.tree_.value
        self.classes = sklearn_tree.classes_
        self.n_classes = sklearn_tree.n_classes_

        if tree_rep is None:
            tree_rep = self.map_tree()
        self.tree_rep = tree_rep

    def map_tree(self):
        class_names = self.classes
        tree_representation = {0: {"depth": 0,
                                   "parent": -1}}
        nodes_to_check = [0]
        while len(nodes_to_check) > 0:
            node = nodes_to_check.pop(0)

            left_child = self.children_left[node]
            tree_representation[node]["left"] = left_child
            if left_child != -1:
                tree_representation[left_child] = {"parent": node,
                                                   "type": "left"}
                nodes_to_check.append(left_child)
            right_child = self.children_right[node]
            tree_representation[node]["right"] = right_child
            if right_child != -1:
                tree_representation[right_child] = {"parent": node,
                                                    "type": "right"}
                nodes_to_check.append(right_child)

            tree_representation[node]["feature"] = self.feature[node]
            tree_representation[node]["threshold"] = self.threshold[node]

            if node != 0:
                parent = tree_representation[node]["parent"]
                tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
                parent_cond = tree_representation[parent]["condition"]
                sign = "<=" if tree_representation[node]["type"] == "left" else ">"
                cond = {
                    "feature": self.feature[parent],
                    "sign": sign,
                    "thresh": self.threshold[parent]
                }
                tree_representation[node]["condition"] = parent_cond + [cond]
            else:  # root
                tree_representation[node]["condition"] = []

            if left_child == -1:  # leaf
                value = self.value[node]
                class_name = np.argmax(value)
                class_name = class_names[class_name]
                tree_representation[node]["class"] = class_name

        return tree_representation

    def predict(self, x_data):  # dataframe (samples, features)
        x_data = x_data.to_numpy()
        prediction = []
        for sample in x_data:
            node = 0
            while self.children_left[node] > 0:  # go in path to relevant leaf
                feature = self.feature[node]
                threshold = self.threshold[node]
                if sample[feature] <= threshold:  # go left
                    node = self.children_left[node]
                else:  # go rights
                    node = self.children_right[node]

            value = self.value[node]
            class_name = np.argmax(value)
            class_name = self.classes[class_name]
            prediction.append(class_name)

        return np.array(prediction)

    def replace_subtree(self, node, subtree):  # subtree is sklearn tree
        n_nodes_orig = self.node_count
        n_classes = self.n_classes

        node_type = self.tree_rep[node]["type"]
        parent = self.tree_rep[node]["parent"]
        if node_type == "left":
            self.children_left[parent] = n_nodes_orig
        else:  # right child
            self.children_right[parent] = n_nodes_orig

        child_left_new = subtree.tree_.children_left + n_nodes_orig
        child_right_new = subtree.tree_.children_right + n_nodes_orig
        self.children_left = np.append(self.children_left, child_left_new)
        self.children_right = np.append(self.children_right, child_right_new)

        self.threshold = np.append(self.threshold, subtree.tree_.threshold)
        self.feature = np.append(self.feature, subtree.tree_.feature)
        # model.tree_.impurity = np.append(model.tree_.impurity, new_subtree.tree_.impurity)
        # model.tree_.n_node_samples = np.append(model.tree_.n_node_samples, new_subtree.tree_.n_node_samples)

        node_depth = self.tree_rep[node]["depth"]
        max_depth = max(self.max_depth, node_depth + subtree.tree_.max_depth)
        self.max_depth = max_depth

        self.node_count += subtree.tree_.node_count

        if subtree.n_classes_ == n_classes and (subtree.classes_ == self.classes).sum() == n_classes:  # same order
            self.value = np.append(self.value, subtree.tree_.value, axis=0)
        else:
            new_values = np.zeros([subtree.tree_.node_count, 1, n_classes])
            for i in range(subtree.n_classes_):
                class_name = subtree.classes_[i]
                c = np.where(self.classes == class_name)[0]
                new_values[:, :, c] = subtree.tree_.value[:, :, i][0]
            self.value = np.append(self.value, new_values, axis=0)

        tree_rep = self.map_tree()
        self.tree_rep = tree_rep

if __name__ == '__main__':
    pass

