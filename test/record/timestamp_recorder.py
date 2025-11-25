import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PropagationRecord:
    """单次传播记录"""
    iteration: int
    phase: str  # 'forward' or 'backward'
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """计算持续时间(秒)"""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def __str__(self) -> str:
        duration = f"{self.duration*1000:.2f}ms" if self.duration else "未完成"
        return f"Iter {self.iteration} | {self.phase:8s} | 耗时: {duration}"


class TimestampRecorder:
    """时间戳记录容器"""
    
    def __init__(self):
        self.records: List[PropagationRecord] = []
        self.current_iteration = 0
        self._forward_start: Optional[float] = None
        self._backward_start: Optional[float] = None
    
    def start_forward(self):
        """记录前向传播开始"""
        self._forward_start = time.time()
    
    def end_forward(self):
        """记录前向传播结束"""
        if self._forward_start is None:
            raise ValueError("前向传播未开始,无法记录结束时间")
        
        record = PropagationRecord(
            iteration=self.current_iteration,
            phase='forward',
            start_time=self._forward_start,
            end_time=time.time()
        )
        self.records.append(record)
        self._forward_start = None
    
    def start_backward(self):
        """记录反向传播开始"""
        self._backward_start = time.time()
    
    def end_backward(self):
        """记录反向传播结束"""
        if self._backward_start is None:
            raise ValueError("反向传播未开始,无法记录结束时间")
        
        record = PropagationRecord(
            iteration=self.current_iteration,
            phase='backward',
            start_time=self._backward_start,
            end_time=time.time()
        )
        self.records.append(record)
        self._backward_start = None
        self.current_iteration += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        forward_times = [r.duration for r in self.records if r.phase == 'forward' and r.duration]
        backward_times = [r.duration for r in self.records if r.phase == 'backward' and r.duration]
        
        stats = {}
        if forward_times:
            stats['forward_avg'] = sum(forward_times) / len(forward_times)
            stats['forward_min'] = min(forward_times)
            stats['forward_max'] = max(forward_times)
        
        if backward_times:
            stats['backward_avg'] = sum(backward_times) / len(backward_times)
            stats['backward_min'] = min(backward_times)
            stats['backward_max'] = max(backward_times)
        
        return stats
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("训练时间统计摘要")
        print("="*60)
        
        stats = self.get_statistics()
        
        if 'forward_avg' in stats:
            print(f"\n前向传播:")
            print(f"  平均耗时: {stats['forward_avg']*1000:.2f}ms")
            print(f"  最小耗时: {stats['forward_min']*1000:.2f}ms")
            print(f"  最大耗时: {stats['forward_max']*1000:.2f}ms")
        
        if 'backward_avg' in stats:
            print(f"\n反向传播:")
            print(f"  平均耗时: {stats['backward_avg']*1000:.2f}ms")
            print(f"  最小耗时: {stats['backward_min']*1000:.2f}ms")
            print(f"  最大耗时: {stats['backward_max']*1000:.2f}ms")
        
        print(f"\n总迭代次数: {self.current_iteration}")
        print("="*60 + "\n")
    
    def print_records(self, last_n: Optional[int] = None):
        """打印记录"""
        records_to_print = self.records[-last_n:] if last_n else self.records
        
        print("\n详细记录:")
        print("-" * 60)
        for record in records_to_print:
            print(record)
        print("-" * 60 + "\n")
    
    def reset(self):
        """重置记录器"""
        self.records.clear()
        self.current_iteration = 0
        self._forward_start = None
        self._backward_start = None
    
    def export_to_dict(self) -> List[Dict]:
        """导出为字典列表,方便保存"""
        return [
            {
                'iteration': r.iteration,
                'phase': r.phase,
                'start_time': r.start_time,
                'end_time': r.end_time,
                'duration': r.duration
            }
            for r in self.records
        ]