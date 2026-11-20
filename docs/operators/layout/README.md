# CPU 张量布局算子

CPU-06 为 `Reshape`、`Permute` 和 `Reorder` 提供 FP32、FP16、BF16 的 prepared reference kernel。三者复制元素字节，保留 NaN payload 和带符号零。

| 算子 | 动态执行合同 |
|---|---|
| `Reshape` | 连续输入与连续输出之间按逻辑顺序复制。绑定时验证 literal、输入轴引用、相等 volume 和唯一推导维。 |
| `Permute` | `permutation[j]` 是输出轴 `j` 对应的输入轴。执行时按轴映射搬移，允许合法的非连续输入和输出。 |
| `Reorder` | 形状和坐标值保持原样，将合法输入步长收拢到连续输出。 |

非空写入要求输出布局可证 injective，输入输出互斥。空张量是无操作。物化由请求绑定提供的输出存储承载。每个 kernel 只保存验证过的地址、形状与步长，执行不分配 scratch。

`Reshape` 的输出可带由输入绑定的符号维。若唯一推导维在 lowering 后仍为 unknown，调用方通过外部输出提供具体形状，执行绑定再验证 volume 与目标。内部 activation 仍需由已有符号绑定求出形状。动态 shape 求解和零拷贝 view 需要独立的 planner 与 lifetime 设计。

语义图的 activation 输入端口接受只读外部张量，保持其 model-input 绑定身份。使用实际 registry 和同一份 lowered plan 测试两种输入长度，覆盖 `Reorder → Reshape → Permute → Cast → Softmax → Cast`。[执行证据](benchmarks/desktop-54h5mmi-layout.md)。
