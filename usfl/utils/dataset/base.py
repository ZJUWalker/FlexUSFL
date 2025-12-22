from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from datasets import disable_progress_bar
from torch.utils.data import DataLoader
from functools import partial
from usfl.utils.dataset.exp import get_dra_test_label, get_dra_train_label
from usfl.utils.dataset.exp import get_dataset
import math
import random


class FedDataset(ABC):
    """
    Federated (Split) Learning Dataset
    支持 'exclusive' (互斥切分) 和 'random_overlap' (随机重叠采样)
    """

    def __init__(
        self,
        tokenizer,
        client_ids: list[str],
        dataset,
        types: list[str],
        shrink_frac=1.0,
        num_labels=0,
        completion_only=False,
        uni_length: int = -1,
        partition_mode: str = "exclusive",  # 选项: "exclusive", "random_overlap"
        sample_ratio: float = 0.1,  # 仅在 random_overlap 模式下生效：每个客户端采样比例
        seed: int = 42,  # 随机种子，保证实验可复现
    ):
        if not client_ids:
            client_ids = [0]

        self.tokenizer = tokenizer
        self.client_ids = client_ids
        self.client_data_indices = {}
        self.all_dataset = dataset
        self.dataset = {}
        self.completion_only = completion_only
        self.num_labels = num_labels
        self.uni_length = uni_length
        random.seed(seed)

        for type in types:
            # 应用 shrink_frac 缩小数据集
            self.dataset[type] = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))

            # 将数据索引分为 len(client_ids) 个子集
            indices = list(range(len(self.dataset[type])))
            total_samples = len(indices)
            num_clients = len(client_ids)

            shards = []

            # choose partition_mode
            if partition_mode == "exclusive":
                shard_size = math.ceil(total_samples / num_clients)
                shards = [indices[i * shard_size : (i + 1) * shard_size] for i in range(num_clients)]
            elif partition_mode == "random_overlap":
                # 随机重叠采样
                num_samples_per_client = int(total_samples * sample_ratio)
                num_samples_per_client = max(1, min(num_samples_per_client, total_samples))

                for _ in range(num_clients):
                    # random.sample 是无放回采样（单个客户端内不重复），但不同客户端之间会重叠
                    client_indices = random.sample(indices, k=num_samples_per_client)
                    shards.append(client_indices)
            else:
                raise ValueError(f"Unsupported partition_mode: {partition_mode}")

            # 为每个客户端分配一个分片
            self.client_data_indices[type] = {client_ids[i]: shards[i] for i in range(num_clients)}

    def _collate_fn(self, x, max_seq_len):
        return self._col_fun(x, max_seq_len=max_seq_len)

    def get_dataloader(self, client_id, batch_size=1, type="train", max_seq_len=-1, shuffle=True):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])
        return DataLoader(
            self._pre_process(ds, batch_size),
            collate_fn=partial(self._collate_fn, max_seq_len=max_seq_len),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def as_dataset_and_collator(self, type="train", shrink_frac=1.0):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
        ds = self._pre_process(ds, 1)
        return ds, partial(self._col_fun, extra_info=False)

    def get_dataloader_unsliced(
        self,
        batch_size=2,
        type="train",
        shrink_frac=1.0,
        further_test_split=None,
        max_seq_len=-1,
        shuffle=True,
    ):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
        if further_test_split is not None:
            ds_split = ds.train_test_split(shuffle=shuffle, test_size=further_test_split)
            return DataLoader(
                self._pre_process(ds_split["train"], batch_size),
                collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
                batch_size=batch_size,
                shuffle=shuffle,
            ), DataLoader(
                self._pre_process(ds_split["test"], batch_size),
                collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
                batch_size=batch_size,
                shuffle=shuffle,
            )
        return DataLoader(
            self._pre_process(ds, batch_size),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
        )

    def _pre_process(self, ds, batch_size):
        ds = ds.map(lambda x: self._format(x), batched=False)
        ds.set_format(type="torch")
        return ds

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b["input"] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {
            "input_ids": input["input_ids"],
            "attention_mask": input["attention_mask"],
            "input_text": texts,
        }

    @abstractmethod
    def _format(self, example):
        raise NotImplementedError


class MixtureFedDataset(FedDataset):

    def _format(self, example):
        pass

    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, dataset_names=None, dataset_classes=None):
        super().__init__(tokenizer, client_ids, None, [], shrink_frac)
        if dataset_names is None:
            dataset_names = []
            dataset_classes = []
        self.fed_datasets = []
        self.dataset_names = dataset_names
        for cls in dataset_classes:
            self.fed_datasets.append(cls(tokenizer, client_ids, shrink_frac))

    def get_dataloader(self, client_id, batch_size=1, type="train", max_seq_len=-1):
        return CombinedDataLoader(*[ds.get_dataloader(client_id, batch_size, type, max_seq_len=max_seq_len) for ds in self.fed_datasets])

    def get_dataloader_unsliced(
        self,
        batch_size=2,
        type=None,
        shrink_frac=1.0,
        further_test_split=None,
        max_seq_len=-1,
        shuffle=True,
    ):
        train_loaders = []
        test_loaders = []
        for nm, ds in zip(self.dataset_names, self.fed_datasets):
            if get_dra_train_label(nm) == get_dra_test_label(nm):
                d1, d2 = ds.get_dataloader_unsliced(
                    batch_size,
                    get_dra_train_label(nm),
                    shrink_frac,
                    further_test_split=0.3,
                    max_seq_len=max_seq_len,
                    shuffle=shuffle,
                )
            else:
                d1 = ds.get_dataloader_unsliced(
                    batch_size,
                    get_dra_train_label(nm),
                    shrink_frac=shrink_frac,
                    max_seq_len=max_seq_len,
                    shuffle=shuffle,
                )
                d2 = ds.get_dataloader_unsliced(
                    batch_size,
                    get_dra_test_label(nm),
                    shrink_frac=shrink_frac,
                    max_seq_len=max_seq_len,
                    shuffle=shuffle,
                )
            train_loaders.append(d1)
            test_loaders.append(d2)
        return CombinedDataLoader(*train_loaders), CombinedDataLoader(*test_loaders)


class CombinedDataLoader:
    """
    Combine multiple DataLoaders into one
    """

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self):
        # list of iterators
        iterators = [iter(dataloader) for dataloader in self.dataloaders]

        while iterators:
            # randomly select a DataLoader
            idx = torch.randint(len(iterators), (1,)).item()
            try:
                # get the next data from the DataLoader
                data = next(iterators[idx])
                data["type"] = idx
                yield data
            except StopIteration:
                # delete the DataLoader if there is no more data
                del iterators[idx]


def lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm):
    """Assign different sample number for each client using Log-Normal distribution.

    Sample numbers for clients are drawn from Log-Normal distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        unbalance_sgm (float): Log-normal variance. When equals to ``0``, the partition is equal to :func:`balance_partition`.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    if unbalance_sgm != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client), sigma=unbalance_sgm, size=num_clients)
        client_sample_nums = (client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

    return client_sample_nums


def random_slicing(dataset, num_clients, sgm=0):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    if num_clients > 0:
        user_samples = lognormal_unbalance_split(num_clients, len(dataset), sgm)
    for i in range(num_clients):
        dict_users[i] = list(np.random.choice(all_idxs, user_samples[i], replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def get_client_dataloaders(
    dataset_name,
    tokenizer,
    client_ids,
    batch_size=2,
    max_seq_len=-1,
    splits=["train", "test"],
    shuffle=False,
    partition_mode="exclusive",
    sample_ratio=0.1,
):
    if not client_ids:
        raise ValueError("客户端 ID 列表不能为空。")

    usl_dataset = get_dataset(
        dataset_name, tokenizer=tokenizer, client_ids=client_ids, partition_mode=partition_mode, sample_ratio=sample_ratio
    )
    client_dataloaders = {client_id: {} for client_id in client_ids}

    has_validation = False if dataset_name in ["codealpaca", "gsm8k"] else True
    has_test = False if dataset_name in ["e2e"] else True

    for split in splits:
        if not has_validation and split == "validation":
            continue
        if not has_test and split == "test":
            continue
        for client_id in client_ids:
            data_loader = usl_dataset.get_dataloader(
                client_id=client_id,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                type=split,
                shuffle=shuffle,
            )
            client_dataloaders[client_id][split] = data_loader

    return client_dataloaders
