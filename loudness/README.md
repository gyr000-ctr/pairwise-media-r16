# loudness

## 题面（可直接粘贴）

子目录：loudness。EBU R128 综合响度。约束：K 加权（高架+高通）系数与标准一致（≤1e-9）；门限（绝对 -70 LUFS 与相对 -10 LU）两级都要实现；1 kHz 0 dBFS 正弦读数与参考差 ≤0.1 LU；静音返回 -inf 而不是 NaN；100 万样本不超过 600 ms；非法参数可区分报错。验收：cd loudness && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd loudness && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
