import pandas
import os
from datasets import load_dataset

def load_data(dataset_name, subset_name, save_path='data/raw'):

    dataset = load_dataset(dataset_name, subset_name)
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        full_path = os.path.join(save_path, dataset_name.replace('/', '_'))
        os.makedirs(full_path, exist_ok=True)
        df.to_csv(f'{full_path}/{subset_name}_{split}.csv', index=False)
        print(f"Saved {split} dataset to {full_path}/{subset_name}_{split}.csv")

if __name__ == "__main__":

    load_data('shmuhammad/AfriSenti-twitter-sentiment', 'ibo')