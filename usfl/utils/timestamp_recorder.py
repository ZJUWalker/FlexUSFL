from dataclasses import asdict, dataclass, field
from typing import List, Dict, Optional


@dataclass
class GanttChartData:
    client_id: Optional[int] = None
    batch_idx: int = 0
    train_time_duration_ms: float = 0.0
    head_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    head_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    head_fwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_fwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_bwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_bwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)

    client_step_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_step_timestamp: List[float] = field(default_factory=lambda: [None] * 2)

    client_fed_avg_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
