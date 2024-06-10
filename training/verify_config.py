import pandas as pd
import configparser
import datasets

RANDOM_STATE = 42


def load_train_test(**kwargs):
    string = f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['train_data']}"
    print(f'path : {string}')
    string = f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['test_data']}"
    print(f'path : {string}')
    train_dataset = pd.read_json(f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['train_data']}", lines=True)
    test_dataset = pd.read_json(f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['test_data']}", lines=True)

    max_data_points = kwargs["max_data_points"]
    if max_data_points != "all":
        max_data_points = int(max_data_points)

        if train_dataset.shape[0] > max_data_points:
            train_dataset = train_dataset.sample(n=int(max_data_points), random_state=RANDOM_STATE)

    train_dataset = datasets.Dataset.from_pandas(train_dataset)
    test_dataset = datasets.Dataset.from_pandas(test_dataset)

    return train_dataset, test_dataset


def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


if __name__ == "__main__":
    DATA_FOLDER = "datasets"
    config_file = "config.ini"
    data_info_file = "dataset_info.ini"
    training_config_file = "training.ini"

    model_configs = read_config(config_file)
    data_configs = read_config(data_info_file)
    for model_config in list(model_configs):
        config = dict(model_configs[model_config].items())
        dataset_list = eval(config["datasets"])
        for n, dataset_name in enumerate(dataset_list):
            print(dataset_name)
            dataset_config = dict(data_configs[dataset_name].items())
            train_dataset, test_dataset = load_train_test(**dataset_config)
