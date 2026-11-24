"""uvunwrap 核心 API（初始未实现）。"""


class Unwrap:
    def unwarp(self, verts, faces, mode='box'):
        """返回每顶点 UV"""
        raise NotImplementedError

    def distortion(self):
        """面积失真"""
        raise NotImplementedError
