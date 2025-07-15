_dataset_name_map = {}
_dataset_dra_train_label_map = {}
_dataset_dra_test_label_map = {}


class AverageMeter(object):  #
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def register_dataset(name, dra_train_label='validation', dra_test_label='test'):
    from usfl.utils.dataset.base import FedDataset

    def wrapper(cls):
        assert issubclass(cls, FedDataset), "All dataset must inherit FedDataset"
        _dataset_name_map[name] = cls
        _dataset_dra_train_label_map[name] = dra_train_label
        _dataset_dra_test_label_map[name] = dra_test_label
        return cls

    return wrapper


def get_dataset(dataset_name, tokenizer, client_ids=None, shrink_frac=1.0, completion_only=False):
    if client_ids is None:
        client_ids = []
    if ',' in dataset_name:
        dataset_names = dataset_name.split(',')
        dataset_classes = [get_dataset_class(dn) for dn in dataset_names]
        from usfl.utils.dataset.base import MixtureFedDataset

        return MixtureFedDataset(tokenizer, client_ids, shrink_frac, dataset_names, dataset_classes)
    else:
        return get_dataset_class(dataset_name)(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=shrink_frac, completion_only=completion_only)


def get_dataset_class(dataset_name):
    from usfl.utils.dataset import datasets

    clz = datasets.FedDataset
    if dataset_name not in _dataset_name_map:
        raise AttributeError
    clz = _dataset_name_map[dataset_name]
    return clz


def get_dra_train_label(name):
    return _dataset_dra_train_label_map[name]


def get_dra_test_label(name):
    return _dataset_dra_train_label_map[name]
