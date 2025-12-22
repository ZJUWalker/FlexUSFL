import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import argparse

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Timeline visualization for server-client data")
parser.add_argument("-V", "--version", type=str, default="v1", help="Version of the configuration")
parser.add_argument("-LAG", "--lag", type=int, default=0, help="Lag value")
parser.add_argument("-NC", "--client_num", type=int, default=3, help="Number of clients")
parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
parser.add_argument("-QO", "--queue_order", type=str, default="fifo", help="queue order for clients")
parser.add_argument("-SB", "--start_batch", type=int, default=1, help="Start batch index")  # ✅ 新增
parser.add_argument("-EB", "--end_batch", type=int, default=None, help="End batch index (None = all)")  # ✅ 新增

args = parser.parse_args()

# 使用命令行参数值
version = args.version
lag = args.lag
client_num = args.client_num
model = args.model.split("/")[-1]
dataset = args.dataset
queue_order = args.queue_order
start_batch = args.start_batch  # ✅ 从命令行参数读取
end_batch = args.end_batch  # ✅ 从命令行参数读取
bps = 2

# 读取数据
dir = f"./version_{version}/model_{model}/dataset_{dataset}/lag_{lag}/client_num_{client_num}/order_{queue_order}"
print(f"Reading data from {dir}")
path = dir + "/server_profile_data_merged.json"
with open(path, "r") as f:
    data = json.load(f)

# ✅ 如果 end_batch 为 None，设置为最大 batch_idx
if end_batch is None:
    max_batch_idx = 0
    for client_id, batches in data.items():
        for batch in batches:
            max_batch_idx = max(max_batch_idx, batch.get("batch_idx", 0))
    end_batch = max_batch_idx
    print(f"未指定 end_batch，自动设置为最大值: {end_batch}")

print(f"绘制范围: Batch {start_batch} ~ {end_batch}")

# 设置图形
fig, ax = plt.subplots(figsize=(40, 7))

# 定义颜色映射
colors = {
    "server_fwd": "#FF6B6B",
    "server_bwd": "#FF6B6B",
    "server_step": "#1DD1A1",
    "head": "#3498DB",
    "tail": "#2ECC71",
    "client_step": "#E67E22",
    "head_fwd_send": "#45B7D1",
    "server_fwd_send": "#96CEB4",
    "tail_bwd_send": "#FFEAA7",
    "server_bwd_send": "#DFE6E9",
    "client_fed_avg_timestamp": "#9B59B6",
}


# 辅助函数：检查时间戳是否有效
def is_valid_timestamp(timestamps):
    """检查时间戳是否有效（非 None 且为列表）"""
    if not isinstance(timestamps, list) or len(timestamps) != 2:
        return False
    return timestamps[0] is not None and timestamps[1] is not None


# ✅ 辅助函数：过滤 batch 范围
def filter_batches(batches, start, end):
    """只保留 start_batch <= batch_idx <= end_batch 的数据"""
    return [b for b in batches if start <= b.get("batch_idx", 0) <= end]


# ✅ 计算时间基准（只考虑指定范围的 batch）
min_time = float("inf")
for client_id, batches in data.items():
    filtered_batches = filter_batches(batches, start_batch, end_batch)
    for batch in filtered_batches:
        for key, value in batch.items():
            if "timestamp" in key and isinstance(value, list):
                valid_values = [v for v in value if v is not None]
                if valid_values:
                    min_time = min(min_time, min(valid_values))

if min_time == float("inf"):
    print("⚠️  警告：指定范围内没有有效的时间戳数据！")
    min_time = 0

# 行索引：server 在第 0 行，clients 从第 1 行开始
row_map = {"server": 0}
client_ids = sorted([int(k) for k in data.keys()])
for i, cid in enumerate(client_ids):
    row_map[f"client_{cid}"] = i + 1

total_rows = len(client_ids) + 1
row_height = 0.8

# 绘制时间线
patches = []
final_filtered_data = {}
for client_id, batches in data.items():
    client_id = int(client_id)
    filtered_batches = filter_batches(batches, start_batch, end_batch)  # ✅ 过滤范围

    for batch in filtered_batches:
        batch_idx = batch["batch_idx"]
        server_row = row_map["server"]
        client_row = row_map[f"client_{client_id}"]

        # ---- Server forward ----
        if "server_fwd_timestamp" in batch:
            timestamps = batch["server_fwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, server_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["server_fwd"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    server_row,
                    f"C{client_id}\nB{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Server backward ----
        if "server_bwd_timestamp" in batch:
            timestamps = batch["server_bwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, server_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor="none",
                    edgecolor=colors["server_bwd"],
                    linewidth=0.5,
                    hatch="///",
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    server_row,
                    f"C{client_id}\nB{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colors["server_bwd"],
                )

        # ---- Server step ----
        if "server_step_timestamp" in batch:
            timestamps = batch["server_step_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, server_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["server_step"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    server_row,
                    f"C{client_id}\nB{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Client head fwd ----
        if "head_fwd_timestamp" in batch:
            timestamps = batch["head_fwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["head"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Client head bwd ----
        if "head_bwd_timestamp" in batch:
            timestamps = batch["head_bwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor="none",
                    edgecolor=colors["head"],
                    linewidth=0.5,
                    hatch="///",
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colors["head"],
                )

        # ---- Client tail fwd ----
        if "tail_fwd_timestamp" in batch:
            timestamps = batch["tail_fwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["tail"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Client tail bwd ----
        if "tail_bwd_timestamp" in batch:
            timestamps = batch["tail_bwd_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor="none",
                    edgecolor=colors["tail"],
                    linewidth=0.5,
                    hatch="///",
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colors["tail"],
                )

        # ---- Client step ----
        if "client_step_timestamp" in batch:
            timestamps = batch["client_step_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["client_step"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Client fed avg timestamp ----
        if "client_fed_avg_timestamp" in batch:
            timestamps = batch["client_fed_avg_timestamp"]
            if is_valid_timestamp(timestamps):
                start, end = timestamps
                start_rel = (start - min_time) * 1000
                duration = (end - start) * 1000
                rect = mpatches.Rectangle(
                    (start_rel, client_row - row_height / 2),
                    duration,
                    row_height,
                    facecolor=colors["client_fed_avg_timestamp"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                patches.append(rect)

                ax.text(
                    start_rel + duration / 2,
                    client_row,
                    f"B{batch_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

        # ---- Send operations ----
        send_ops = ["head_fwd_send", "server_fwd_send", "tail_bwd_send", "server_bwd_send"]
        for op in send_ops:
            key = f"{op}_timestamp"
            if key in batch:
                timestamps = batch[key]
                if is_valid_timestamp(timestamps):
                    start, end = timestamps
                    start_rel = (start - min_time) * 1000
                    duration = (end - start) * 1000
                    rect = mpatches.Rectangle(
                        (start_rel, client_row - row_height / 2),
                        duration,
                        row_height,
                        facecolor=colors[op],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    patches.append(rect)

                    ax.text(
                        start_rel + duration / 2,
                        client_row,
                        f"B{batch_idx}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

# 添加所有 patches
for patch in patches:
    ax.add_patch(patch)

# 设置坐标轴
ax.set_ylim(-0.5, total_rows - 0.5)
ax.set_yticks(range(total_rows))
ax.set_yticklabels(["Server"] + [f"Client {cid}" for cid in client_ids])
ax.invert_yaxis()

ax.set_xlabel("Time (ms)", fontsize=12)
ax.grid(axis="x", alpha=0.3)

# ✅ 计算并展示最右结束时间（只考虑指定范围的 batch）
max_time = 0
for client_id, batches in data.items():
    filtered_batches = filter_batches(batches, start_batch, end_batch)
    for batch in filtered_batches:
        for key, value in batch.items():
            if "timestamp" in key and isinstance(value, list):
                valid_values = [v for v in value if v is not None]
                if valid_values:
                    max_time = max(max_time, max(valid_values))

total_duration_ms = (max_time - min_time) * 1000
ax.set_xlim(0, total_duration_ms)

# 在图右下角标注终点时间
ax.text(
    1.0,
    -0.08,
    f"End: {total_duration_ms:.1f} ms",
    # f"End: {total_duration_ms:.1f} ms\n(Batch {start_batch}-{end_batch})",  # ✅ 显示范围
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12,
)

print(f"最右边结束时间: {total_duration_ms:.1f} ms")

# ---- 图例 ----
legend_elements = [
    mpatches.Patch(facecolor=colors["server_fwd"], edgecolor="black", label="Server Forward"),
    mpatches.Patch(facecolor="none", edgecolor=colors["server_bwd"], label="Server Backward", hatch="///"),
    mpatches.Patch(facecolor=colors["server_step"], edgecolor="black", label="Server Step"),
    mpatches.Patch(facecolor=colors["head"], edgecolor="black", label="Head Forward"),
    mpatches.Patch(facecolor="none", edgecolor=colors["head"], label="Head Backward", hatch="///"),
    mpatches.Patch(facecolor=colors["tail"], edgecolor="black", label="Tail Forward"),
    mpatches.Patch(facecolor="none", edgecolor=colors["tail"], label="Tail Backward", hatch="///"),
    mpatches.Patch(facecolor=colors["client_step"], edgecolor="black", label="Client Step"),
    mpatches.Patch(facecolor=colors["head_fwd_send"], edgecolor="black", label="Head Fwd Send"),
    mpatches.Patch(facecolor=colors["server_fwd_send"], edgecolor="black", label="Server Fwd Send"),
    mpatches.Patch(facecolor=colors["tail_bwd_send"], edgecolor="black", label="Tail Bwd Send"),
    mpatches.Patch(facecolor=colors["server_bwd_send"], edgecolor="black", label="Server Bwd Send"),
]
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    fontsize=14,
    ncol=6,
    frameon=False,
)

plt.tight_layout()
savepath = dir + f"/training_timeline.png"  # ✅ 文件名包含范围
plt.savefig(savepath, dpi=300, bbox_inches="tight")
plt.show()

print(f"时间线可视化已保存为 {savepath}")
