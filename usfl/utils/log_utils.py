import os

import logging

from usfl import env


def create_logger(
    log_file_name: str = "default.log",
    log_dir: str = env.log_dir,
    log_level: str = "INFO",
    format: str = "%(asctime)s %(levelname)s %(message)s",
    mode: str = "a",
    console_output: bool = True,
) -> logging.Logger:
    """
    设置单独的logger

    参数:
        log_file_name: 日志文件名
        log_dir: 日志目录
        log_level: 日志级别（INFO, DEBUG, WARNING, ERROR, CRITICAL）
        format: 日志格式
        mode: 文件写入模式（'a' 追加 / 'w' 覆盖）
        console_output: 是否在终端输出日志（默认True）
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file_path = os.path.join(log_dir, log_file_name)

    # 创建 Logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(log_level)

    # 创建文件处理器并设置输出的文件
    file_handler = logging.FileHandler(log_file_path, mode=mode)
    file_handler.setLevel(log_level)

    # 创建日志格式器并设置格式
    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    # 添加文件处理器到 Logger
    logger.addHandler(file_handler)

    # 如果 console_output=True，则添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 防止日志被向上（propagate）传递
    logger.propagate = False

    return logger
