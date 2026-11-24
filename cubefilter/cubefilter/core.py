"""cubefilter 核心 API（初始未实现）。"""


class CubeFilter:
    def convolve(self, env, size):
        """漫反射卷积"""
        raise NotImplementedError

    def sample(self, cube, face, u, v):
        """采样"""
        raise NotImplementedError
