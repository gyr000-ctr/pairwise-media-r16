"""loudness 核心 API（初始未实现）。"""


class Loudness:
    def __init__(self, rate):
        """构造"""
        raise NotImplementedError

    def integrated(self, samples):
        """综合响度"""
        raise NotImplementedError

    def k_weight(self, samples):
        """K 加权输出"""
        raise NotImplementedError
