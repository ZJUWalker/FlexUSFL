# 将模型拆分成三份，head，server，tail
import os
import json
from dataclasses import dataclass, field, asdict


@dataclass
class SplitModelConfig:
    head_layer_num: int = field(default=2)
    server_layer_num: int = field(default=-1)  # 如果不设置会在拆分时自动根据头尾层数计算
    tail_layer_num: int = field(default=2)
    with_server: bool = field(default=True)
    logicl_load: bool = field(default=True)

    @property
    def total_hidden_layers(self):
        return self.head_layer_num + self.server_layer_num + self.tail_layer_num

    def save_pretrained(self, save_dir: str):
        pth = os.path.join(save_dir, 'split_config.json')
        with open(pth, 'w') as f:
            f.write(json.dumps(asdict(self)))

    @staticmethod
    def load_pretrained(save_dir: str):
        pth = os.path.join(save_dir, 'split_config.json')
        with open(pth, 'r') as f:
            config_dict = json.load(f)
        return SplitModelConfig(**config_dict)
