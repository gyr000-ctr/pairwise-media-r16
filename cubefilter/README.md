# cubefilter

## 题面（可直接粘贴）

子目录：cubefilter。立方体贴图漫反射卷积。约束：卷积结果与球面均匀采样对照（≤1e-3）；常量环境卷积后仍为常量（能量守恒）；面间接缝连续；非法分辨率可区分报错；64x64x6 不超过 800 ms；并发安全。验收：cd cubefilter && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd cubefilter && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
