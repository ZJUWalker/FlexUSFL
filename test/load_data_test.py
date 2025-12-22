import torch
import sys
import numpy as np
from transformers import AutoTokenizer
from usfl.utils.load_utils import load_client, load_dataset

# === 修改这里：导入你实际的 load_dataset 函数 ===
# 假设你的代码在当前目录下的 data_loader.py 中
# from data_loader import load_dataset
# 如果代码就在同一个文件中，或者你可以直接粘贴到这里运行
# 为了演示，我假设你已经能正确导入 load_dataset
# try:
#     from main import load_dataset  # 请根据你的实际文件名修改导入路径
# except ImportError:
#     print("错误: 请修改脚本开头的 import 语句，指向包含 load_dataset 的文件。")
#     # sys.exit(1)


def test_exclusive_mode(tokenizer, dataset_name="gsm8k"):
    print("\n" + "=" * 50)
    print("测试模式 1: Exclusive (互斥切分)")
    print("=" * 50)

    num_clients = 4
    client_ids = list(range(num_clients))

    # 调用加载函数
    dataloaders = load_dataset(
        dataset_name=dataset_name, tokenizer=tokenizer, client_ids=client_ids, batch_size=2, partition_mode="exclusive"  # <--- 关键参数
    )

    # 验证数据
    all_data_fingerprints = []
    client_data_sets = []

    for cid in client_ids:
        loader = dataloaders[cid]["train"]
        client_samples = []

        # 遍历 DataLoader 获取所有数据指纹 (这里用 input_text 做指纹)
        for batch in loader:
            # batch['input_text'] 是我们在 _col_fun 里返回的原文列表
            client_samples.extend(batch["input_text"])

        count = len(client_samples)
        print(f"[Client {cid}] 数据量: {count} 条")

        # 存入集合以便比对
        client_set = set(client_samples)
        client_data_sets.append(client_set)
        all_data_fingerprints.extend(client_samples)

    # 验证互斥性
    c0_set = client_data_sets[0]
    c1_set = client_data_sets[1]
    intersection = c0_set.intersection(c1_set)

    print(f"\n检查 Client 0 和 Client 1 的交集: {len(intersection)} 条重复")
    if len(intersection) == 0:
        print("✅ 验证通过: 客户端之间无数据重叠。")
    else:
        print("❌ 验证失败: 发现重叠数据 (预期应为0)。")


def test_random_overlap_mode(tokenizer, dataset_name="gsm8k"):
    print("\n" + "=" * 50)
    print("测试模式 2: Random Overlap (随机重叠)")
    print("=" * 50)

    num_clients = 3
    client_ids = list(range(num_clients))
    ratio = 0.5  # 让每个客户端拥有 50% 的数据，必然会发生重叠

    # 调用加载函数
    dataloaders = load_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        client_ids=client_ids,
        batch_size=2,
        partition_mode="random_overlap",  # <--- 关键参数
        sample_ratio=ratio,  # <--- 关键参数
    )

    client_data_sets = []

    for cid in client_ids:
        loader = dataloaders[cid]["train"]
        client_samples = []
        for batch in loader:
            client_samples.extend(batch["input_text"])

        count = len(client_samples)
        print(f"[Client {cid}] 数据量: {count} 条 (目标比例 {ratio})")
        client_data_sets.append(set(client_samples))

    # 验证重叠性
    c0_set = client_data_sets[0]
    c1_set = client_data_sets[1]
    intersection = c0_set.intersection(c1_set)

    print(f"\n检查 Client 0 和 Client 1 的交集: {len(intersection)} 条重复")

    if len(intersection) > 0:
        print(f"✅ 验证通过: 发现 {len(intersection)} 条重叠数据 (符合预期)。")
    else:
        print("⚠️ 警告: 未发现重叠数据 (如果 sample_ratio 很小或是运气极好可能发生，但 0.5 比例下几乎不可能)。")


if __name__ == "__main__":
    # 1. 初始化 Tokenizer (这里使用一个轻量级的，或者你可以 Mock 一个)
    # 如果你本地没有 gpt2，可以换成任何你有的模型，或者 mock
    try:
        print("正在加载 Tokenizer...")
        model_dir = "/share/models/meta-llama/llama3.2-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        print("尝试使用 Mock Tokenizer...")

        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # 返回伪造的 token
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            def pad(self, *args, **kwargs):
                return {}

        tokenizer = MockTokenizer()

    # 2. 运行测试
    # 注意：确保你的网络能连接 HuggingFace 加载 gsm8k 数据集，
    # 或者 dataset_cache_dir 里已经有缓存
    try:
        test_exclusive_mode(tokenizer)
        # test_random_overlap_mode(tokenizer)
    except Exception as e:
        print(f"\n❌ 测试运行出错: {e}")
        import traceback

        traceback.print_exc()
