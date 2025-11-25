import torch
import torch.nn as nn
import torch.optim as optim
from timestamp_recorder import TimestampRecorder


def create_model():
    """创建简单的神经网络模型"""
    return nn.Sequential(
        nn.Linear(100, 500),
        nn.ReLU(),
        nn.Linear(500, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )


def train(model, optimizer, criterion, recorder, num_epochs=5, batch_size=32):
    """训练函数"""
    print(f"开始训练 {num_epochs} 个 epoch...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # 生成随机数据 (模拟真实数据)
        x = torch.randn(batch_size, 100)
        y = torch.randint(0, 10, (batch_size,))
        
        # 前向传播
        recorder.start_forward()
        output = model(x)
        loss = criterion(output, y)
        recorder.end_forward()
        
        # 反向传播
        recorder.start_backward()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recorder.end_backward()
        
        # 打印当前进度
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")
    
    print("-" * 60)


def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模型、优化器和损失函数
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 创建时间戳记录器
    recorder = TimestampRecorder()
    
    # 训练模型
    train(model, optimizer, criterion, recorder, num_epochs=10)
    
    # 打印最后5条记录
    print("\n最近5次迭代的详细记录:")
    recorder.print_records(last_n=5)
    
    # 打印统计摘要
    recorder.print_summary()
    
    # 导出数据 (可选)
    # import json
    # data = recorder.export_to_dict()
    # with open('training_log.json', 'w') as f:
    #     json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()