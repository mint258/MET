# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

# 数据列表
datasets = ['FreeSolv', 'Lipo', 'QM7', 'QM9']
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
    'Our_model'
]

# 处理每个数据集的RMSE值和标准差
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

# 动态计算每个数据集的基准RMSE值（均值，忽略NaN）
benchmark_values = {}
for dataset in datasets:
    means = np.array(data[dataset]['means'])
    # 使用np.nanmean计算平均值，忽略NaN
    if np.isnan(means).all():
        # 如果所有值都是NaN，设置基准值为0或其他默认值
        benchmark = 0
    else:
        benchmark = np.nanmean(means)
    benchmark_values[dataset] = benchmark
    
# 初音未来配色
miku_main = "#39C5BB"  # 主色（青绿色）
miku_accent = "#FF69B4"  # 辅助色（粉色）
miku_background = "#F0F8FF"  # 背景色
miku_font = "#004F59"  # 深青色字体

# 创建一个大图，包含4个子图（2行2列）
fig, axes = plt.subplots(2, 2, figsize=(24, 18))
axes = axes.flatten()  # 将二维数组展平成一维，便于迭代

# 设置背景色
fig.patch.set_facecolor(miku_background)

# 设置每个子图的位置和数据
for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    means = data[dataset]['means']
    stds = data[dataset]['stds']
    
    # 处理缺失数据（NaN）
    means = np.array(means)
    stds = np.array(stds)
    
    # 创建一个布尔掩码，排除NaN值
    valid = ~np.isnan(means)
    valid_means = means[valid]
    valid_stds = stds[valid]
    valid_models = np.array(models)[valid]
    
    # 设置X轴位置
    x = np.arange(len(valid_models))
    
    # 绘制柱状图
    bars = ax.bar(x, valid_means, yerr=np.nan_to_num(valid_stds, nan=0), color=miku_main, align='center', 
                  alpha=0.9, ecolor=miku_accent, capsize=5, edgecolor='black')
    
    # 绘制基准线
    benchmark = benchmark_values.get(dataset, None)
    if benchmark is not None:
        ax.axhline(y=benchmark, color='red', linestyle='--', linewidth=2, label='Benchmark')
        ax.legend()
    
    # 设置X轴标签和刻度
    ax.set_xlabel('Models', fontsize=14, color=miku_font)
    ax.set_ylabel('RMSE', fontsize=14, color=miku_font)
    ax.set_title(f'RMSE on {dataset} Dataset', fontsize=16, color=miku_font)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45, ha='right', fontsize=12, color=miku_font)
    ax.tick_params(axis='y', colors=miku_font)
    
    # 动态调整y轴上限以适应文本标签
    # 使用 np.nan_to_num 处理 stds 中的 NaN，确保不会引入 NaN
    combined = valid_means + np.nan_to_num(valid_stds, nan=0)
    
    if combined.size > 0:
        max_combined = np.max(combined)
    else:
        max_combined = 1  # 默认值，防止 max_y 为空
    
    # 如果存在基准值，比较并取最大值
    if benchmark is not None:
        max_y = max(max_combined, benchmark)
    else:
        max_y = max_combined
    
    # 检查 max_y 是否为有效数值
    if not np.isfinite(max_y):
        max_y = 1  # 设置一个默认值
    
    ax.set_ylim(0, max_y * 1.2)
    
    # 添加数值标签，避免覆盖误差条
    for bar, mean, std in zip(bars, valid_means, valid_stds):
        height = bar.get_height()
        # 如果 std 是 NaN，则不添加 std
        if not np.isnan(std):
            label_y = height + std + 0.02 * max_y
            label_text = f'{mean:.3f}\n({std:.3f})'
        else:
            label_y = height + 0.02 * max_y
            label_text = f'{mean:.3f}'
        
        # 确保标签不会超出 y 轴范围
        label_y = min(label_y, max_y * 1.1)
        
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                label_text,
                ha='center', va='bottom', fontsize=10, color=miku_font)

# 调整布局，增加子图间距
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 保存图片
plt.savefig('benchmark_result_miku_style_updated.png', dpi=300, facecolor=miku_background)

# 输出目录，确保存在
output_dir = 'benchmark_individual_plots'
os.makedirs(output_dir, exist_ok=True)

# 定义一个函数来绘制单个数据集的图形
def plot_individual_benchmark(dataset, means, stds, benchmark, models, colors, output_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 设置背景色
    fig.patch.set_facecolor(colors['background'])
    ax.set_facecolor(colors['background'])
    
    # 处理缺失数据（NaN）
    means = np.array(means)
    stds = np.array(stds)
    
    # 创建一个布尔掩码，排除NaN值
    valid = ~np.isnan(means)
    valid_means = means[valid]
    valid_stds = stds[valid]
    valid_models = np.array(models)[valid]
    
    # 设置X轴位置
    x = np.arange(len(valid_models))
    
    # 绘制柱状图，处理 stds 中的 NaN
    bars = ax.bar(x, valid_means, yerr=np.nan_to_num(valid_stds, nan=0), color=colors['main'], align='center', 
                  alpha=0.9, ecolor=colors['accent'], capsize=5, edgecolor='black')
    
    # 绘制基准线
    if benchmark is not None:
        ax.axhline(y=benchmark, color='red', linestyle='--', linewidth=2, label='Benchmark')
        ax.legend()
    
    # 设置X轴标签和刻度
    ax.set_xlabel('Models', fontsize=14, color=colors['font'])
    ax.set_ylabel('RMSE', fontsize=14, color=colors['font'])
    ax.set_title(f'RMSE on {dataset} Dataset', fontsize=16, color=colors['font'])
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45, ha='right', fontsize=12, color=colors['font'])
    ax.tick_params(axis='y', colors=colors['font'])
    
    # 动态调整y轴上限以适应文本标签
    combined = valid_means + np.nan_to_num(valid_stds, nan=0)
    
    if combined.size > 0:
        max_combined = np.max(combined)
    else:
        max_combined = 1  # 默认值，防止 max_combined 为空
    
    # 如果存在基准值，比较并取最大值
    if benchmark is not None:
        max_y = max(max_combined, benchmark)
    else:
        max_y = max_combined
    
    # 检查 max_y 是否为有效数值
    if not np.isfinite(max_y):
        max_y = 1  # 设置一个默认值
    
    ax.set_ylim(0, max_y * 1.2)
    
    # 添加数值标签，避免覆盖误差条
    for bar, mean, std in zip(bars, valid_means, valid_stds):
        height = bar.get_height()
        # 如果 std 是 NaN，则不添加 std
        if not np.isnan(std):
            label_y = height + std + 0.02 * max_y
            label_text = f'{mean:.3f}\n({std:.3f})'
        else:
            label_y = height + 0.02 * max_y
            label_text = f'{mean:.3f}'
        
        # 确保标签不会超出 y 轴范围
        label_y = min(label_y, max_y * 1.1)
        
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                label_text,
                ha='center', va='bottom', fontsize=10, color=colors['font'])
    
    # 保存图片
    plt.savefig(output_path, dpi=300, facecolor=colors['background'])
    plt.close()

# 遍历每个数据集，创建单独的图形
for dataset in datasets:
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
            'main': miku_main,
            'accent': miku_accent,
            'background': miku_background,
            'font': miku_font
        },
        output_path=output_path
    )
    
    print(f'Saved plot for {dataset} to {output_path}')

print('所有基准测试图形已成功创建并保存。')