import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 创建一个用于写入日志文件的处理器
    file_handler = logging.FileHandler('log/0707.log')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    # 创建一个用于终端显示的处理器
    # stream_handler = logging.StreamHandler()
    # stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S')
    # stream_handler.setFormatter(stream_formatter)
    # 将两个处理器添加到日志记录器
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    return logger