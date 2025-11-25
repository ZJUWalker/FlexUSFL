import random
import numpy as np
import torch

# 所有关于路径的拼接都必须要使用os.path.join()，不允许使用+，防止路径拼接错误
project_root = '/home/lzh/projects/FlexUSFL'
nltk_path = '~/nltk_data'  # 用于计算meteor指标数据集
dataset_cache_dir = '/share/datasets/'
model_path = f'{project_root}/data/models'
model_save_path = f'{project_root}/data/ft_models'
naive_train_model_save_path = f'{model_save_path}/naive'
sl_model_save_path = f'{model_save_path}/sl'
log_dir = f'{project_root}/log'

train_log_dir = f'{log_dir}/train'


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
