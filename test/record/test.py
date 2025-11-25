import time
import torch
import torch.nn as nn
import torch.optim as optim


class BigMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=4096, num_layers=10, num_classes=1000):
        """
        一个比较大的多层全连接网络：
        input_dim -> hidden_dim x (num_layers-1) -> num_classes
        """
        super().__init__()
        layers = []

        # 第一层：input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # 中间隐藏层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层：hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sync(device):
    """
    在 GPU 上需要同步才能得到准确计时；CPU 上则什么都不做。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_one_iter(model, criterion, optimizer, x, target, device):
    """
    测一个 iteration（forward + loss + backward + step）的时间，
    并单独拆出各部分开销（单位：毫秒）。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # ==== forward ====
    sync(device)
    t0 = time.perf_counter()
    out = model(x)
    sync(device)
    t1 = time.perf_counter()

    # ==== loss compute ====
    sync(device)
    loss = criterion(out, target)
    sync(device)
    t2 = time.perf_counter()

    # ==== backward ====
    sync(device)
    loss.backward()
    sync(device)
    t3 = time.perf_counter()

    # ==== optimizer step ====
    sync(device)
    optimizer.step()
    sync(device)
    t4 = time.perf_counter()

    forward_time = (t1 - t0) * 1000
    loss_time = (t2 - t1) * 1000
    backward_time = (t3 - t2) * 1000
    step_time = (t4 - t3) * 1000
    total_time = (t4 - t0) * 1000

    return forward_time, loss_time, backward_time, step_time, total_time


def main():
    # ====================== 配置参数 ======================
    batch_size = 512        # 批大小（可以改大/改小来测试）
    input_dim = 2048        # 输入特征维度
    hidden_dim = 4096       # 每层隐藏单元数
    num_layers = 10         # 全连接层总层数（包含第一层和最后一层）
    num_classes = 1000      # 输出类别数
    lr = 1e-3               # 学习率
    warmup_iters = 5        # 预热迭代次数
    measure_iters = 20      # 正式计时迭代次数（会统计平均）
    # ====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构造大模型
    model = BigMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    # 随机输入 & 标签
    x = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ====================== warmup ======================
    print(f"Warmup for {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        benchmark_one_iter(model, criterion, optimizer, x, target, device)
    print("Warmup done.\n")

    # ====================== 正式计时 ======================
    forward_times = []
    loss_times = []
    backward_times = []
    step_times = []
    total_times = []

    print(f"Measuring {measure_iters} iterations...")
    for i in range(measure_iters):
        fwd, l, bwd, step, total = benchmark_one_iter(
            model, criterion, optimizer, x, target, device
        )
        forward_times.append(fwd)
        loss_times.append(l)
        backward_times.append(bwd)
        step_times.append(step)
        total_times.append(total)

        print(
            f"Iter {i+1:02d}: "
            f"forward={fwd:.3f} ms, "
            f"loss={l:.3f} ms, "
            f"backward={bwd:.3f} ms, "
            f"step={step:.3f} ms, "
            f"total={total:.3f} ms"
        )

    # ====================== 统计平均 ======================
    import statistics

    def avg(lst):
        return statistics.mean(lst) if len(lst) > 0 else 0.0

    print("\n====== AVERAGE OVER {0} ITERS ======".format(measure_iters))
    print(f"Forward avg:      {avg(forward_times):.3f} ms")
    print(f"Loss compute avg: {avg(loss_times):.3f} ms")
    print(f"Backward avg:     {avg(backward_times):.3f} ms")
    print(f"Step avg:         {avg(step_times):.3f} ms")
    print(f"Total avg:        {avg(total_times):.3f} ms")


if __name__ == "__main__":
    main()
