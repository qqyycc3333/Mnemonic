import os
from typing import List, Dict

from datasets import load_dataset, concatenate_datasets

from collections import Counter
import json

PATH = "/home/zwy478049/workspace/aaawin/LLaMA-Factory/data"      # modify


def load_and_save(ds_name, save_name, mode='all'):
    """
    mode: 'all', 'sample'
    """
    ds = load_dataset(ds_name)
    print(ds)

    if save_name == "metamathqa":
        ds = ds['train']
        ratio = 0.1
        counter = Counter(ds['type'])
        new_dataset = []
        for key, value in counter.items():
            new_dataset.append(ds.filter(lambda x: x['type']==key).shuffle().select(range(int(value*ratio))))

        new_dataset = concatenate_datasets(new_dataset)
        ds: List[Dict[str, str]] = [data for data in new_dataset]
        print(len(ds))

        os.makedirs(os.path.join(PATH, save_name), exist_ok=True)
        with open(os.path.join(PATH, save_name, f"{save_name}_train.json"), 'w') as f:
            json.dump(ds, f, indent=2)
    
    # Not Used.
    if save_name == "medmcqa":
        if mode == 'sample':
            ds = ds['train']
            # 过滤掉长文本和缺乏explanation的样本
            ds_filter = ds.filter(lambda x: \
                                        20 < len(x['instruction'].split(" ")) < 100 and 
                                        20 < len(x['output'].split(" ")) < 100)
            ds_shuffle = ds_filter.remove_columns(["input"]).shuffle()
            # 只保留答案
            # def extract_word(batch):
            #     output = batch['output'].split(" ")[2][0]
            #     batch['output'] = output
            #     return batch
            # ds_shuffle = ds_shuffle.map(extract_word)
            # print(set(ds_shuffle['output']))
            print(ds_shuffle)

            ds_train = [item for item in ds_shuffle.select(range(40000))]
            ds_test = [item for item in ds_shuffle.select(range(40000, 45000))]

            with open(os.path.join(PATH, save_name, f"{save_name}_train.json"), 'w') as f:
                json.dump(ds_train, f, indent=2)
            with open(os.path.join(PATH, save_name, f"{save_name}_test.json"), 'w') as f:
                json.dump(ds_test, f, indent=2)

        if mode == 'all':
            ds_train = [item for item in ds['train']]
            ds_test = [item for item in ds['validation']]
            print("ds_train", len(ds_train))
            print("ds_test", len(ds_test))

            with open(os.path.join(PATH, save_name, f"{save_name}_train.json"), 'w') as f:
                json.dump(ds_train, f, indent=2)
            with open(os.path.join(PATH, save_name, f"{save_name}_test.json"), 'w') as f:
                json.dump(ds_test, f, indent=2)

    if save_name == "codetester":
        ds = ds['train']
        ds = ds.remove_columns("input")
        ds_filter = ds.filter(lambda x: len(x['instruction'].split(" ")) < 256 and len(x['output'].split(" ")) < 512)
        print("ds_filter", ds_filter)

        ds_filter = [item for item in ds_filter]

        os.makedirs(os.path.join(PATH, save_name), exist_ok=True)
        with open(os.path.join(PATH, save_name, f"{save_name}_train.json"), 'w') as f:
            json.dump(ds_filter, f, indent=2)

if __name__ == "__main__":

    # ds_name = "meta-math/MetaMathQA"
    # save_name = 'metamathqa'

    # ds_name = "chenhaodev/medmcqa_instruct"
    # save_name = 'medmcqa'

    ds_name = "Vezora/Tested-143k-Python-Alpaca"
    save_name = 'codetester'

    # ds_name = "gsm8k"
    load_and_save(ds_name, save_name, "all")