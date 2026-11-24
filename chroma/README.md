# chroma

## 题面（可直接粘贴）

子目录：chroma。色度子采样（4:2:0）。约束：亮度通道逐像素不变；色度恢复与双线性上采样一致（≤1e-9）；往返（下采样+上采样）PSNR ≥40 dB；非法尺寸可区分报错；100 万像素不超过 500 ms；并发安全。验收：cd chroma && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd chroma && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
