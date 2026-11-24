# uvunwrap

## 题面（可直接粘贴）

子目录：uvunwrap。盒式/平面 UV 展开。约束：展开后 UV 落在 [0,1]；面积失真 ≤5%（与 3D 面积比）；接缝数量最少（盒子法 ≤6 组）；非法模式可区分报错；1 万三角面不超过 500 ms；并发安全。验收：cd uvunwrap && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd uvunwrap && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
