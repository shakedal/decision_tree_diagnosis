import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from sklearn import metrics

from DataSet import DataSet
from ResultsToExcel import write_to_excel
from buildModel import build_model, map_tree, prune_tree
from updateModel import print_tree_rules
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints
import random
import copy

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
epsilon = np.finfo(np.float64).eps

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

change_types = {
    "half_std_up": 0.5,
    "half_std_down": -0.5,
    "one_std_up": 1,
    "one_std_down": -1,
    "two_std_up": 2,
    "two_std_down": -2
}

severity = {
    "half_std_up": 1,
    "half_std_down": 1,
    "one_std_up": 2,
    "one_std_down": 2,
    "two_std_up": 3,
    "two_std_down": 3,
    "0.3": 1,
    "0.5": 2,
    "0.7": 3,
    "0.9": 4
    # "0.1": 1,
    # "0.2": 2,
    # "0.3": 3
    # "uniform dist": 2,
    # "original dist": 2,
    # "filtered dist": 1,
    # "softmax orig dist": 2,
    # "softmax filtered dist": 1
}

def change_data_numeric(feature, all_data, filtered_data, indexes_filtered_data):
    # calculating statistics
    mean = float(all_data.mean()[feature])
    std = float(all_data.std()[feature])

    half_std_up = filtered_data.loc[indexes_filtered_data, feature] + 0.5 * std
    half_std_down = filtered_data.loc[indexes_filtered_data, feature] - 0.5 * std
    one_std_up = filtered_data.loc[indexes_filtered_data, feature] + 1 * std
    one_std_down = filtered_data.loc[indexes_filtered_data, feature] - 1 * std
    two_std_up = filtered_data.loc[indexes_filtered_data, feature] + 2 * std
    two_std_down = filtered_data.loc[indexes_filtered_data, feature] - 2 * std
    feature_changes = [half_std_up, half_std_down, one_std_up, one_std_down, two_std_up, two_std_down]
    feature_changes_names = ["half_std_up", "half_std_down", "one_std_up", "one_std_down", "two_std_up",
                             "two_std_down"]

    return feature_changes, feature_changes_names

def change_data_binary(feature, all_data, filtered_data, indexes_filtered_data):
    values = all_data[feature].unique()
    value_counts = all_data[feature].value_counts()
    rows_to_change = indexes_filtered_data.sum()
    random.seed(17)
    uniform_dist = random.choices(values, weights=None, k=rows_to_change)

    distribution = np.zeros(len(values))
    for i in range(len(values)):
        val = values[i]
        distribution[i] = value_counts[val]
    distribution /= len(all_data)
    random.seed(5)
    orig_dist = random.choices(values, weights=distribution, k=rows_to_change)

    distribution3 = softmax(distribution)
    random.seed(31)
    softmax_orig_dist = random.choices(values, weights=distribution3, k=rows_to_change)

    values2 = filtered_data.loc[indexes_filtered_data, feature].unique()
    value_counts2 = filtered_data.loc[indexes_filtered_data, feature].value_counts()
    distribution2 = np.zeros(len(values2))
    for i in range(len(values2)):
        val = values2[i]
        distribution2[i] = value_counts2[val]
    distribution2 /= rows_to_change
    random.seed(13)
    filtered_dist = random.choices(values2, weights=distribution2, k=rows_to_change)

    distribution4 = softmax(distribution2)
    random.seed(7)
    softmax_filtered_dist = random.choices(values2, weights=distribution4, k=rows_to_change)

    feature_changes = [uniform_dist, orig_dist, filtered_dist, softmax_orig_dist, softmax_filtered_dist]
    feature_changes_names = ["uniform dist", "original dist", "filtered dist", "softmax orig dist",
                             "softmax filtered dist"]

    return feature_changes, feature_changes_names

def change_data_binary2(feature, all_data, filtered_data, indexes_filtered_data):
    values = all_data[feature].unique()
    feature_changes = []
    feature_changes_names = []
    changes = [0.3, 0.5, 0.7, 0.9]
    original = filtered_data.copy()

    for val in values:
        to_change = original.copy()
        indexes = to_change[feature] != val
        n = indexes.values.sum()
        indexes = to_change[indexes].index.values
        if n < 1:
            continue
        for p in changes:
            to_change = original.copy()
            random.seed(10)
            indexes_to_change = random.choices(indexes, k=int(n*p))
            to_change.loc[indexes_to_change, feature] = val
            feature_changes.append(to_change[feature])
            feature_changes_names.append(f"{p}")

    # feature_changes = [uniform_dist, orig_dist, filtered_dist, softmax_orig_dist, softmax_filtered_dist]
    # feature_changes_names = ["uniform dist", "original dist", "filtered dist", "softmax orig dist",
    #                          "softmax filtered dist"]

    return feature_changes, feature_changes_names

def manipulate_feature(feature, feature_type, dataset):
    concept_size = dataset.before_size
    not_changed_data = dataset.data.iloc[0:concept_size].copy()
    to_change_data = dataset.data.iloc[concept_size:].copy()
    indexes_to_change = (to_change_data[feature] > 0) | (to_change_data[feature] <= 0)  # all indexes

    all_data = dataset.data
    # creating changes
    if feature_type == "numeric":
        feature_changes, feature_changes_names = change_data_numeric(feature, all_data, to_change_data, indexes_to_change)
    else:  # binary \ categorical
        feature_changes, feature_changes_names = change_data_binary2(feature, all_data, to_change_data, indexes_to_change)

    # saving changes to csv \ yield
    for i in range(len(feature_changes)):
        data_change = feature_changes[i]
        change_name = feature_changes_names[i]
        to_save = to_change_data.copy()
        to_save.loc[indexes_to_change, feature] = data_change
        to_save = not_changed_data.append(to_save, ignore_index=True)
        yield to_save, change_name

def simulate_drift(feature, feature_type, dataset):
    # order indexes by feature value
    sorted_data = dataset.data.sort_values(by=feature)

    # divide to before & after
    n_samples = sorted_data.shape[0]
    n_before = dataset.before_size
    n_after = n_samples - n_before
    indexes_before_drift = list(range(n_before))
    indexes_after_drift = list(range(n_before, n_samples))

    # sample before
    random.seed(42)
    indexes_before = random.sample(indexes_before_drift, k=int(0.8*n_before))
    random.seed(13)
    indexes_before += random.sample(indexes_after_drift, k=int(0.2*n_before))
    assert len(indexes_before) == len(set(indexes_before))

    # sample after + test
    indexes_after = list(filter(lambda x: x not in indexes_before, range(n_samples)))
    assert len(indexes_before) + len(indexes_after) == n_samples

    # combine to one df
    all_indexes = indexes_before + indexes_after
    drift_data = sorted_data.iloc[all_indexes]
    drift_data = drift_data.reset_index(drop=True)

    return drift_data

def is_feature_in_tree(tree_rep, feature_num):
    for node, d in tree_rep.items():
        if d["feature"] == feature_num:
            return True
    return False

all_sizes = [
    (0.7, 0.1, 0.2),
    (0.7, 0.07, 0.2),
    (0.7, 0.05, 0.2),
    (0.7, 0.02, 0.2),
    (0.7, 0.01, 0.2),
    (0.7, 0.005, 0.2)
]

# all_sizes = [
#     (0.6, 0.2, 0.2),
#     (0.6, 0.1, 0.2),
#     (0.6, 0.07, 0.2),
#     (0.6, 0.05, 0.2),
#     (0.6, 0.02, 0.2),
#     (0.6, 0.01, 0.2),
#     (0.6, 0.005, 0.2)
# ]

if __name__ == '__main__':
    all_results = []
    time_stamp = datetime.now()
    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")

    all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)

    categorical_datasets = ["annealing"]
    # categorical_datasets = ["analcatdata_boxing1", "braziltourism", "meta", "newton_hema", "socmob", "vote", "newton_hema", "visualizing_livestock"]

    for index, row in all_datasets.iterrows():
        # if row["name"] not in categorical_datasets:
        #     continue
        # if index > 10:  # use for testing
        #     break

        print(f'------------------DATASET: {row["name"]}------------------')
        data_size = row["dataset size"]

        dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, (0.7, 0.1, 0.2), name=row["name"],
                          to_shuffle=True)
        # build tree
        concept_size = dataset.before_size
        target = dataset.target
        feature_types = dataset.feature_types
        train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
        validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
        model = build_model(train, dataset.features, dataset.target, val_data=validation)
        tree_rep = map_tree(model)
        model = prune_tree(model, tree_rep)
        print("TREE:")
        print_tree_rules(model, dataset.features)
        tree_rep = map_tree(model)

        # check performances without drift - after set
        performances = {}

        for sizes in all_sizes:
            # check if there is enough data for experiment
            if sizes[1] * data_size < 1:
                continue

            dataset1 = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, sizes,
                              name=row["name"], to_shuffle=True)
            after_samples = dataset1.data.iloc[concept_size:concept_size + dataset1.after_size].copy()
            after_data_x = after_samples[dataset1.features]
            prediction = model.predict(after_data_x)
            after_data_y = after_samples[dataset1.target]
            accuracy_after_no_drift = metrics.accuracy_score(after_data_y, prediction)
            performances[sizes] = accuracy_after_no_drift

        # check performances without drift - test set
        test_samples = dataset.data.iloc[len(dataset.data) - dataset.test_size: -1].copy()
        test_data_x = test_samples[dataset.features]
        prediction = model.predict(test_data_x)
        test_data_y = test_samples[dataset.target]
        accuracy_test_no_drift = metrics.accuracy_score(test_data_y, prediction)

        data_before_manipulation = dataset.data.iloc[0:concept_size]

        # manipulate data & run experiment
        for i in range(len(dataset.features)):
            feature = dataset.features[i]
            feature_type = dataset.feature_types[i]
            if feature_type == "categorical":
                n_values = dataset.data[feature].unique().size
                if n_values > 2:
                    f_type = "categorical"
                else: f_type = "binary"
            else:
                f_type = "numeric"
            if not is_feature_in_tree(tree_rep, i):
                continue
            manipulated_data = manipulate_feature(feature, feature_type, dataset)

            for data, change in manipulated_data:
                for sizes in all_sizes:
                    # check if there is enough data for experiment
                    if sizes[1] * data_size < 1:
                        continue

                    dataset_for_exp = DataSet(data, "diagnosis_check", target, feature_types, sizes)
                    data_after_manipulation = dataset_for_exp.data.iloc[0:concept_size]
                    assert data_after_manipulation.equals(data_before_manipulation.astype(data_after_manipulation.dtypes)), \
                        f"before:\n{data_before_manipulation}\n\nafter:\n{data_after_manipulation}"

                    with HiddenPrints():
                        result = run_single_tree_experiment(dataset_for_exp, model=copy.deepcopy(model),
                                                            check_diagnosis=False, faulty_nodes=[i])
                    result["size"] = sizes[1]
                    result["dataset"] = dataset.name
                    result["change severity"] = severity[change]
                    if change in change_types:
                        change_type = change_types[change]
                        result["change type"] = change_type
                    else: result["change type"] = change
                    result["feature type"] = feature_type
                    result["f type"] = f_type
                    # result["is feature in tree?"] = is_feature_in_tree(tree_rep, i)
                    result["number of faulty nodes"] = 1
                    result["model accuracy - no drift - after"] = performances[sizes]
                    result["model accuracy - no drift - test"] = accuracy_test_no_drift
                    all_results.append(result)

    write_to_excel(all_results, f"ALL_FEATURE_DIFF_FIX_result_run_{date_time}")

    print("DONE")

# if __name__ == '__main__':
#     all_results = []
#     time_stamp = datetime.now()
#     date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")
#
#     all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)
#     for index, row in all_datasets.iterrows():
#         # if index > 3:  # use for testing
#         #     break
#
#         print(f'------------------DATASET: {row["name"]}------------------')
#         data_size = row["dataset size"]
#
#         dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, (0.7, 0.1, 0.2),
#                           name=row["name"], to_shuffle=False)
#
#         # manipulate data & run experiment
#         for i in range(len(dataset.features)):
#             feature = dataset.features[i]
#             feature_type = dataset.feature_types[i]
#             manipulated_data = simulate_drift(feature, feature_type, dataset)
#             dataset = DataSet(manipulated_data.copy(), "diagnosis_check", None, None, (0.7, 0.1, 0.2),
#                               name=row["name"], to_shuffle=False)
#
#             # build tree
#             concept_size = dataset.before_size
#             target = dataset.target
#             feature_types = dataset.feature_types
#             train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
#             validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
#             model = build_model(train, dataset.features, dataset.target, val_data=validation)
#             tree_rep = map_tree(model)
#             model = prune_tree(model, tree_rep)
#             print("TREE:")
#             print_tree_rules(model, dataset.features)
#             tree_rep = map_tree(model)
#
#             # pass if feature not in tree
#             if not is_feature_in_tree(tree_rep, i):
#                 continue
#
#             # experiments
#             for sizes in all_sizes:
#                 # check if there is enough data for experiment
#                 if sizes[1] * data_size < 1:
#                     continue
#
#                 dataset1 = DataSet(manipulated_data.copy(), "diagnosis_check", None, None, sizes,
#                                   name=row["name"], to_shuffle=False)
#                 with HiddenPrints():
#                     result = run_single_tree_experiment(dataset1, model=copy.deepcopy(model),
#                                                         check_diagnosis=False, faulty_nodes=[i])
#
#                 result["size"] = sizes[1]
#                 result["dataset"] = dataset.name
#                 result["feature type"] = feature_type
#                 result["number of faulty nodes"] = 1
#                 all_results.append(result)
#
#     write_to_excel(all_results, f"result_run_{date_time}")
#
#     print("DONE")
