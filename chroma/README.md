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

初始 15 failed；实现后 15 条规格测试全部通过。`tests/test_spec_acceptance.py`
另含 56 条对照验收约束的回归测试。

## 实现说明

API：`chroma.Chroma`（`split` / `merge` / `subsample` / `upsample`），模块级同名
函数为便捷包装。数据格式统一为 `[[(r, g, b), ...], ...]` 嵌套列表（也接受 NumPy
数组），亮度/色度平面为 `[[v, ...], ...]`。

- **颜色矩阵**：BT.601 全范围（JFIF）。`Y = 0.299R + 0.587G + 0.114B`，
  `Cb = (B-Y)/1.772 + 0.5`，`Cr = (R-Y)/1.402 + 0.5`。亮度逐像素等于上式
  （与输入浮点运算完全一致），矩阵可逆，因此 `merge(split(rgb))` 误差在 1e-12 内。
- **下采样**：2x2 盒式平均；奇数边以边缘复制补齐，输出 `ceil(H/2) x ceil(W/2)`。
- **上采样**：像素中心对齐（align_corners=False）的可分离双线性插值，边界复制；
  与独立参考实现逐元素一致（≤1e-12）。`merge` 收到的半分辨率色度即按此恢复，
  因此与「先双线性上采样、再逆矩阵重建」的参考流程一致（≤1e-9）。
  `merge` 同时接受 ceil 与 floor 两种半分辨率尺寸。
- **色域**：`merge` 输出裁剪到 `[0,1]`（只在颜色越界处生效；合法范围内
  `merge(split())` 精确还原）。
- **往返 PSNR**：对平滑自然的图像（渐变/正弦/径向/平滑噪声）可达 55–80 dB；
  单像素棋盘等超出 4:2:0 奈奎斯特极限的图样本质上无法承载，不在 ≥40 dB 保证内。
- **错误分类**：`ChromaTypeError`（类型/非数值，同时是 `TypeError` 与
  `ValueError`）、`ChromaShapeError`（空/维度/尺寸不匹配）、`ChromaValueError`
  （NaN/Inf），三者均继承 `ValueError` 并共享基类 `ChromaError`。
- **并发**：无全局可变状态、不修改输入，所有方法为纯函数式，多线程共享同一实例安全。
- **性能**：100 万像素 `split + subsample×2 + merge` 端到端约 300–420 ms
  （numpy 向量化 + C 层摊平/输出；逐像素解释器循环已避免）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖（运行时仅需 NumPy）
