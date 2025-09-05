import matplotlib.pyplot as plt
import numpy as np

# 设置全局风格（Nature 期刊风格）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "grey"
plt.rcParams["grid.linestyle"] = ""
plt.rcParams["grid.linewidth"] = 0.5

# 数据（保持顺序不变，原来的顺序即：0 freezing layers对应全层可训练，9 freezing layers对应仅最后一层可训练）
trainable_parameters = [7016705, 5160707, 4143363, 3126019, 2108675, 1091331, 1025539, 959747, 893955, 861059]
val_R2 = [0.093334, 0.051507, 0.045321, 0.024821, 0.392669, 0.449262, 0.493894, 0.450495, 0.336593, 0.320349]

# 新的 x 轴：x 值代表可训练层数，取值从10（全层可训练）到1（仅最后一层可训练）
x = list(range(10, 0, -1))

# 对应的层名称（按照网络顺序，从底层到顶层，即当冻结层数不同，可训练部分的最底层名称）
tick_labels = ["Embedding", "EGNN1", "EGNN2", "EGNN3", "EGNN4", "Linear1", "Linear1", "Linear1", "Encoder", "Transformer"]

# 定义背景颜色：EGNN层与 transformer层按照要求指定，其它层自由选择
# 我们设定每个 x 区间 [x-0.5, x+0.5] 的背景色
bg_colors = {
    "Embedding": "#F0F0F0",        # 自由选择的颜色
    "EGNN1": "#9DBB61",         # 指定EGNN层颜色
    "EGNN2": "#9DBB61",
    "EGNN3": "#9DBB61",
    "EGNN4": "#9DBB61",
    "Linear1": "#FFF2CC",         # 自由选择的颜色
    "Linear1": "#FFF2CC",
    "Linear1": "#FFF2CC",
    "Encoder": "#CCE5FF",     # 自由选择的颜色
    "Transformer": "#AB9AC0"  # 指定transformer层颜色
}

# 定义 Nature 风格配色（保留原来的线条颜色）
line1_color = "#1f77b4"  # 蓝色（Trainable Parameters）
line2_color = "#d62728"  # 红色（Validation R²）
background_color = "white"
font_color = "black"

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=background_color)

# 设置 x 轴显示可训练层数
ax1.set_xlabel('Trainable Layers', fontsize=12, color=font_color)
ax1.set_ylabel('Trainable Parameters', fontsize=12, color=line1_color)

# 在主坐标轴上绘制 Trainable Parameters 曲线
line1, = ax1.plot(x, trainable_parameters, color=line1_color, marker='o', linestyle='-', label='Trainable Parameters')
ax1.tick_params(axis='y', colors=line1_color)
ax1.set_xticks(x)
ax1.set_xticklabels(tick_labels, fontsize=10, color=font_color)

ax1.grid(False)

# 为每个 x 区间添加背景色
for xi, label in zip(x, tick_labels):
    ax1.axvspan(xi - 0.5, xi + 0.5, facecolor=bg_colors[label], alpha=0.3)

# 创建共享 x 轴的第二个 y 坐标轴
ax2 = ax1.twinx()
ax2.set_ylabel('Validation R²', fontsize=12, color=line2_color)
line2, = ax2.plot(x, val_R2, color=line2_color, marker='s', linestyle='--', label='Validation R²')
ax2.tick_params(axis='y', colors=line2_color)
ax2.set_ylim(0, 1)

ax2.grid(False)

plt.title('Performance of Fine-tuning with Different Trainable Layers', fontsize=14, color=font_color)

# 添加图例（放在左上角）
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10, frameon=True)

ax1.grid(True, which='both')

# # 添加说明文字（可根据需要调整或删除）
# notice_text = 'Notice: The result of 0 Freezing Layers is obtained by training from scratch with the same structure'
# plt.figtext(0.99, 0.01, notice_text, horizontalalignment='right', fontsize=10, color='gray', alpha=0.7)

fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('freeze_layer.png', dpi=300, facecolor=background_color)
plt.close()
