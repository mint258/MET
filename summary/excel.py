import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局风格（Nature 期刊风格）
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "grey"
plt.rcParams["grid.linestyle"] = ""
plt.rcParams["grid.linewidth"] = 0.5

# 原始数据集顺序
datasets_original = ['FreeSolv', 'Lipo', 'QM7', 'QM9']
# 根据要求重新排序：使 QM9 对应 a，QM7 对应 b，其它依次为 c、d
datasets_order = ['QM9', 'QM7', 'Lipo', 'FreeSolv']

models = [
    'D-MPNN',
    'Attentive FP',
    'N-Gram_RF',
    'N-Gram_XGB',
    'PretrainGNN',
    'GROVER_base',
    'GROVER_large',
    'GraphMVP',
    'GEM',
    'Uni-Mol',
    'MET'
]

data = {
    'FreeSolv': {
        'means': [2.082, 2.073, 2.688, 5.061, 2.764, 2.176, 2.272, np.nan, 1.877, 1.480, 0.462],
        'stds': [0.082, 0.183, 0.085, 0.744, 0.002, 0.052, 0.051, np.nan, 0.094, 0.048, 0.097]
    },
    'Lipo': {
        'means': [0.683, 0.721, 0.812, 2.072, 0.739, 0.817, 0.823, 0.681, 0.660, 0.603, 0.940],
        'stds': [0.016, 0.001, 0.028, 0.030, 0.003, 0.008, 0.010, 0.010, 0.008, 0.010, 0.034]
    },
    'QM7': {
        'means': [103.5, 72.0, 92.8, 81.9, 113.2, 94.5, 92.0, np.nan, 58.9, 41.8, 35.3],
        'stds': [8.6, 2.7, 4.0, 1.9, 0.6, 3.8, 0.9, np.nan, 0.8, 0.2, 5.8]
    },
    'QM9': {
        'means': [0.00814, 0.00812, 0.01037, 0.00964, 0.00922, 0.00984, 0.00986, np.nan, 0.00746, 0.00467, 0.00324],
        'stds': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00004, np.nan]
    }
}

# 计算每个数据集的基准 RMSE 值（忽略 NaN）
benchmark_values = {}
# for dataset in datasets_original:
#     means = np.array(data[dataset]['means'])
#     if np.isnan(means).all():
#         benchmark = 0
#     else:
#         benchmark = np.nanmean(means)
#     benchmark_values[dataset] = benchmark

# 为 QM7、Lipo、FreeSolv 这三个数据集加入单位（QM9 不添加）
unit_mapping = {
    "QM7": " (kcal/mol)",
    "Lipo": " (logP)",
    "FreeSolv": " (kcal/mol)"
}

# 定义 x 轴标题替换映射：dataset -> 对应字母
xlabel_mapping = {
    "QM9": "a",
    "QM7": "b",
    "Lipo": "c",
    "FreeSolv": "d"
}

# Nature 风格配色
bar_color = "#1f77b4"   # 蓝色
error_color = "#ff7f0e" # 橙色
background_color = "white"
font_color = "black"

# 创建 4 行 1 列的子图，整体尺寸设置为 16:9 比例 (16, 9)
fig, axes = plt.subplots(len(datasets_order), 1, figsize=(16, 9), facecolor=background_color)
fig.patch.set_facecolor(background_color)

for ax, dataset in zip(axes, datasets_order):
    means = np.array(data[dataset]['means'])
    stds = np.array(data[dataset]['stds'])
    valid = ~np.isnan(means)
    valid_means = means[valid]
    valid_stds = stds[valid]
    valid_models = np.array(models)[valid]
    x = np.arange(len(valid_models))
    
    # 绘制柱状图（保留原有模型名称作为 x 轴刻度标签）
    bars = ax.bar(x, valid_means, yerr=np.nan_to_num(valid_stds, nan=0),
              color=bar_color, align='center', alpha=0.9, ecolor=error_color,
              capsize=5, edgecolor='black', width=0.5, linewidth=1.5)
    
    # 绘制基准线（如果有）
    benchmark = benchmark_values.get(dataset, None)
    if benchmark is not None:
        ax.axhline(y=benchmark, color='red', linestyle='--', linewidth=2, label='Benchmark')
        ax.legend(fontsize=11)
    
    # 设置 y 轴标签，附加单位（如果有）
    unit = unit_mapping.get(dataset, "")
    ax.set_ylabel('RMSE' + unit, fontsize=12, color=font_color)
    
    # 设置 x 轴标题为映射后的字母，同时增大 labelpad 以免被遮挡
    ax.set_xlabel(xlabel_mapping.get(dataset, "Models"), fontsize=12, color=font_color, labelpad=20)
    
    # 保留原有的 x 轴刻度标签（模型名称），设置为水平显示
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=0, fontsize=11, color=font_color)
    ax.tick_params(axis='y', colors=font_color, labelsize=11)
    
    # 去除每个子图上方的标题
    ax.set_title("")
    
    # 动态调整 y 轴上限
    combined = valid_means + np.nan_to_num(valid_stds, nan=0)
    max_combined = np.max(combined) if combined.size > 0 else 1
    max_y = max(max_combined, benchmark) if benchmark is not None else max_combined
    if not np.isfinite(max_y):
        max_y = 1
    ax.set_ylim(0, max_y * 1.2)
    
    # 为每个柱子添加数值标签
    for bar, mean, std in zip(bars, valid_means, valid_stds):
        height = bar.get_height()
        if not np.isnan(std):
            label_y = height + std + 0.02 * max_y
            label_text = f'{mean:.3f}\n({std:.3f})'
        else:
            label_y = height + 0.02 * max_y
            label_text = f'{mean:.3f}'
        label_y = min(label_y, max_y * 1.1)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                label_text,
                ha='center', va='bottom', fontsize=11, color=font_color)

plt.subplots_adjust(hspace=0.7)
plt.savefig('benchmark_result.png', dpi=300,
            facecolor=background_color, bbox_inches='tight', pad_inches=0)
plt.close()

# 同时输出各数据集的单独图形（下面的调整与整体图类似）
output_dir = 'benchmark_individual_plots'
os.makedirs(output_dir, exist_ok=True)

def plot_individual_benchmark(dataset, means, stds, benchmark, models, colors, output_path):
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=colors['background'])
    fig.patch.set_facecolor(colors['background'])
    ax.set_facecolor(colors['background'])
    
    means = np.array(means)
    stds = np.array(stds)
    valid = ~np.isnan(means)
    valid_means = means[valid]
    valid_stds = stds[valid]
    valid_models = np.array(models)[valid]
    x = np.arange(len(valid_models))
    
    bars = ax.bar(x, valid_means, yerr=np.nan_to_num(valid_stds, nan=0), color=colors['main'],
                  align='center', alpha=0.9, ecolor=colors['accent'], capsize=5, edgecolor='black')
    
    if benchmark is not None:
        ax.axhline(y=benchmark, color='red', linestyle='--', linewidth=2, label='Benchmark')
        ax.legend(fontsize=11)
    
    unit = unit_mapping.get(dataset, "")
    ax.set_xlabel(xlabel_mapping.get(dataset, "Models"), fontsize=12, color=colors['font'], labelpad=20)
    ax.set_ylabel('RMSE' + unit, fontsize=12, color=colors['font'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=0, fontsize=11, color=colors['font'])
    ax.tick_params(axis='y', colors=colors['font'], labelsize=11)
    
    combined = valid_means + np.nan_to_num(valid_stds, nan=0)
    max_combined = np.max(combined) if combined.size > 0 else 1
    max_y = max(max_combined, benchmark) if benchmark is not None else max_combined
    if not np.isfinite(max_y):
        max_y = 1
    ax.set_ylim(0, max_y * 1.2)
    
    for bar, mean, std in zip(bars, valid_means, valid_stds):
        height = bar.get_height()
        if not np.isnan(std):
            label_y = height + std + 0.02 * max_y
            label_text = f'{mean:.3f}\n({std:.3f})'
        else:
            label_y = height + 0.02 * max_y
            label_text = f'{mean:.3f}'
        label_y = min(label_y, max_y * 1.1)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                label_text,
                ha='center', va='bottom', fontsize=11, color=colors['font'])
    
    plt.savefig(output_path, dpi=300, facecolor=colors['background'])
    plt.close()

for dataset in datasets_order:
    means = data[dataset]['means']
    stds = data[dataset]['stds']
    benchmark = benchmark_values.get(dataset, None)
    
    output_filename = f'benchmark_{dataset}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    plot_individual_benchmark(
        dataset=dataset,
        means=means,
        stds=stds,
        benchmark=benchmark,
        models=models,
        colors={
            'main': bar_color,
            'accent': error_color,
            'background': background_color,
            'font': font_color
        },
        output_path=output_path
    )
    print(f'Saved plot for {dataset} to {output_path}')

print('All benchmark plots have been created and saved.')
