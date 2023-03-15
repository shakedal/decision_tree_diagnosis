import pandas as pd
from scipy.io import arff

class DataSet:

    def __init__(self, source_path, dataset_type, target_class, feature_types, size, name=None, to_shuffle=False):
        # check data format
        if type(source_path) == str:
            self.name = source_path
            source_format = source_path.split(".")[-1]
            if source_format in ("csv", "data", "txt"):
                data_df = pd.read_csv(source_path)
            elif source_format == "arff":
                data, meta = arff.loadarff(source_path)
                pd_df = pd.DataFrame(data)
                # pd_df[target_class] = pd_df[target_class].astype('int')
                data_df = pd_df
        else:  # already dataframe
            data_df = source_path
        if name is not None:
            self.name = name

        feature_type = []
        for col in data_df:
            target = col
            if data_df[col].dtype == object:
                feature_type.append("categorical")
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna(data_df[col].mode().iloc[0], inplace=True)
                # convert categorical to nums
                data_df[col] = pd.Categorical(data_df[col])
                data_df[col] = data_df[col].cat.codes
            else:
                feature_type.append("numeric")
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna((data_df[col].mean()), inplace=True)

        if to_shuffle:  # shuffle data, same shuffle always
            data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = data_df

        if target_class is None:
            target_class = target
        self.dataset_type = dataset_type
        self.features = list(self.data.columns)
        self.features.remove(target_class)
        self.target = target_class

        if feature_types is None:
            feature_types = feature_type[:-1]
        self.feature_types = feature_types

        n_samples = len(data_df)

        if type(size) == tuple:
            before_size, after_size, test_size = size
            self.before_size = int(before_size*n_samples)
            self.after_size = int(after_size*n_samples)
            self.test_size = int(test_size*n_samples)

        elif type(size) == int:
            self.before_size = size
            self.after_size = int(size * 0.1)
            self.test_size = int(size * 0.2)

        elif type(size) == list:
            if len(size) == 4:
                concept_size, window, n_used, test = size
            elif len(size) == 5:
                concept_size, window, n_used, test, slot = size
                self.slot = slot
            self.before_size = concept_size - int(window/2)  # "clean" concept
            self.after_size = int(window*n_used)
            self.test_size = int(window*test)
            self.window = window
            self.concept_size = concept_size

        assert (self.before_size + self.after_size + self.test_size) <= n_samples

if __name__ == '__main__':
    pass