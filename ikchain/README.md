# ikchain

## 题面（可直接粘贴）

子目录：ikchain。两骨骼 IK 解析解。约束：末端与目标距离 ≤1e-9（目标在可达域内）；不可达时伸直且方向正确；肘部朝向由 hint 决定（与 hint 点积为正）；关节角连续无跳变；非法骨骼长度可区分报错；10 万次不超过 200 ms；并发安全。验收：cd ikchain && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd ikchain && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
