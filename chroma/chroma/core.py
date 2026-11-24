"""chroma 核心 API（初始未实现）。"""


class Chroma:
    def split(self, rgb):
        """Y/Cb/Cr"""
        raise NotImplementedError

    def merge(self, y, cb, cr):
        """回 RGB"""
        raise NotImplementedError

    def subsample(self, c):
        """下采样"""
        raise NotImplementedError
