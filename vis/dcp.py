import argparse
import json
import os
import re
import glob
from copy import deepcopy
from typing import Any, Dict, List


# 判空工具：None、[]、[None, None] 都视为“空”
def is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, list):
        return len(v) == 0 or all(x is None for x in v)
    return False


# 仅当 server 空且 client 非空时，才从 client 填充到 server
def fill_field(server_val: Any, client_val: Any):
    if is_nullish(server_val) and not is_nullish(client_val):
        return deepcopy(client_val), True
    return server_val, False


def detect_client_id(client_list: List[Dict[str, Any]], filename: str) -> int:
    # 优先从数据里读 client_id
    if client_list and isinstance(client_list[0], dict) and "client_id" in client_list[0]:
        return int(client_list[0]["client_id"])
    # 否则从文件名里解析 client_#_
    m = re.search(r"client_(\d+)_", os.path.basename(filename))
    if not m:
        raise ValueError(f"无法从文件名解析 client_id: {filename}")
    return int(m.group(1))


def merge_one_client(server: Dict[str, Any], client_id: int, client_list: List[Dict[str, Any]]) -> int:
    """
    将 client_list 合并进 server 的该 client（按 batch_idx 对齐）。
    返回填充成功的字段数量。
    """
    key = str(client_id)
    if key not in server or not isinstance(server[key], list):
        raise ValueError(f'server_profile_data.json 中缺少 key "{key}" 或其不是列表')

    server_list = server[key]
    client_by_batch = {item["batch_idx"]: item for item in client_list if "batch_idx" in item}
    filled_count = 0

    for i, s_item in enumerate(server_list):
        bidx = s_item.get("batch_idx")
        c_item = client_by_batch.get(bidx)
        if not c_item:
            continue  # 该 batch 在 client 文件中不存在，跳过

        # 只在 server 字段为空且 client 字段非空时填充
        for k, s_val in list(s_item.items()):
            c_val = c_item.get(k, None)
            new_val, did_fill = fill_field(s_val, c_val)
            if did_fill:
                s_item[k] = new_val
                filled_count += 1

        # 如果 client 有 server 没有的字段，也补上（通常用于结构演进）
        for k, c_val in c_item.items():
            if k not in s_item and not is_nullish(c_val):
                s_item[k] = deepcopy(c_val)
                filled_count += 1

        server_list[i] = s_item

    return filled_count


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Merge server and client data")
    parser.add_argument("-V", "--version", type=str, default="v1", help="Version of the configuration")
    parser.add_argument("-LAG", "--lag", type=int, default=0, help="Lag value")
    parser.add_argument("-NC", "--client_num", type=int, default=3, help="Number of clients")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-QO", "--queue_order", type=str, default="fifo", help="queue order for clients")

    args = parser.parse_args()

    version = args.version
    lag = args.lag
    client_num = args.client_num
    model = args.model.split("/")[-1]
    dataset = args.dataset
    queue_order = args.queue_order
    bps = 2
    # 路径按需修改
    dir = f"./version_{version}/model_{model}/dataset_{dataset}/lag_{lag}/client_num_{client_num}/order_{queue_order}"
    print(f"合并路径: {dir}")
    SERVER_JSON = os.path.join(dir, "server_profile_data.json")
    OUTPUT_JSON = os.path.join(dir, "server_profile_data_merged.json")  # 合并后的输出
    BACKUP_JSON = os.path.join(dir, "server_profile_data.backup.json")  # 备份
    CLIENT_PATTERN = os.path.join(dir, "client_*_profile_data.json")  # 自动发现 client_0/1/2 的文件

    # 读取 server
    with open(SERVER_JSON, "r", encoding="utf-8") as f:
        server = json.load(f)

    # 备份（只备一次）
    if not os.path.exists(BACKUP_JSON):
        with open(BACKUP_JSON, "w", encoding="utf-8") as f:
            json.dump(server, f, ensure_ascii=False, indent=2)

    client_files = sorted(glob.glob(CLIENT_PATTERN))
    if not client_files:
        raise FileNotFoundError(f"未找到任何匹配 {CLIENT_PATTERN} 的客户端文件")

    total_filled = 0
    per_client_filled = {}

    for cf in client_files:
        with open(cf, "r", encoding="utf-8") as f:
            client_list = json.load(f)

        cid = detect_client_id(client_list, cf)
        filled = merge_one_client(server, cid, client_list)
        per_client_filled[cid] = per_client_filled.get(cid, 0) + filled
        total_filled += filled
        print(f"✔ 合并 {cf} -> 客户端 {cid}：填充字段 {filled} 项")

    # 输出合并结果
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(server, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 合并完成，总计填充 {total_filled} 项字段")
    for cid in sorted(per_client_filled):
        print(f"  - 客户端 {cid}: 填充 {per_client_filled[cid]} 项")
    print(f"🛟 备份文件: {BACKUP_JSON}")
    print(f"📄 合并输出: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
