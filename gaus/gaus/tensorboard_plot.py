#!/usr/bin/env python3
"""
TensorBoard日志读取与Loss曲线绘制工具
用于在无UI的Ubuntu环境下读取TensorBoard日志并生成loss曲线图
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于无GUI环境
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

# 设置matplotlib中文字体（可选）
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

def read_tensorboard_logs(log_dir):
    """读取TensorBoard日志文件"""
    print(f"正在读取日志目录: {log_dir}")
    
    # 初始化EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()  # 加载日志数据
    
    print("可用的标量标签:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")
    
    return ea

def extract_scalar_data(ea, tag):
    """提取指定标签的标量数据"""
    try:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        return steps, values
    except KeyError:
        print(f"警告: 标签 '{tag}' 不存在")
        return [], []

def plot_loss_curves(ea, output_dir="./plots"):
    """绘制loss曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    # 首先检查实际可用的标签
    available_tags = ea.Tags()['scalars']
    print(f"\n实际可用的标量标签: {len(available_tags)}个")
    for tag in sorted(available_tags):
        print(f"  - {tag}")
    
    # 定义要绘制的loss指标
    loss_metrics = {
        'train_loss_patches/l1_loss': 'Training L1 Loss',
        'train_loss_patches/total_loss': 'Training Total Loss',
        'test/loss_viewpoint - l1_loss': 'Test L1 Loss',
        'train/loss_viewpoint - l1_loss': 'Train Validation L1 Loss',
        'test/loss_viewpoint - psnr': 'Test PSNR',
        'train/loss_viewpoint - psnr': 'Train Validation PSNR',
        'iter_time': 'Iteration Time (ms)',
        'total_points': 'Total Gaussian Points'
    }
    
    # 1. 绘制训练损失
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('3D Gaussian Splatting Training Metrics', fontsize=16, fontweight='bold')
    
    # L1 Loss
    steps, values = extract_scalar_data(ea, 'train_loss_patches/l1_loss')
    if steps:
        axes[0, 0].plot(steps, values, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training L1 Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('L1 Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Total Loss
    steps, values = extract_scalar_data(ea, 'train_loss_patches/total_loss')
    if steps:
        axes[0, 1].plot(steps, values, 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Training Total Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Total Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Iteration Time
    steps, values = extract_scalar_data(ea, 'iter_time')
    if steps:
        axes[1, 0].plot(steps, values, 'g-', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Iteration Time')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Total Points
    steps, values = extract_scalar_data(ea, 'total_points')
    if steps:
        axes[1, 1].plot(steps, values, 'm-', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Total Gaussian Points')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Number of Points')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"训练指标图已保存到: {os.path.join(output_dir, 'training_metrics.png')}")
    
    # 2. 绘制训练vs验证损失对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training vs Validation Metrics', fontsize=16, fontweight='bold')
    
    # L1 Loss对比 (训练损失 vs 验证损失)
    train_l1_steps, train_l1_values = extract_scalar_data(ea, 'train_loss_patches/l1_loss')
    test_l1_steps, test_l1_values = extract_scalar_data(ea, 'test/loss_viewpoint - l1_loss')
    train_val_l1_steps, train_val_l1_values = extract_scalar_data(ea, 'train/loss_viewpoint - l1_loss')
    
    if train_l1_steps:
        axes[0, 0].plot(train_l1_steps, train_l1_values, 'b-', label='Training L1 Loss', 
                       linewidth=1, alpha=0.7)
    if test_l1_steps:
        axes[0, 0].plot(test_l1_steps, test_l1_values, 'ro-', label='Test L1 Loss', 
                       linewidth=2, markersize=4)
    if train_val_l1_steps:
        axes[0, 0].plot(train_val_l1_steps, train_val_l1_values, 'go-', label='Train Val L1 Loss', 
                       linewidth=2, markersize=4)
    
    axes[0, 0].set_title('L1 Loss: Training vs Validation')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('L1 Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # 使用对数坐标更好地显示loss下降
    
    # PSNR对比
    test_psnr_steps, test_psnr_values = extract_scalar_data(ea, 'test/loss_viewpoint - psnr')
    train_psnr_steps, train_psnr_values = extract_scalar_data(ea, 'train/loss_viewpoint - psnr')
    
    if test_psnr_steps:
        axes[0, 1].plot(test_psnr_steps, test_psnr_values, 'ro-', label='Test PSNR', 
                       linewidth=2, markersize=4)
    if train_psnr_steps:
        axes[0, 1].plot(train_psnr_steps, train_psnr_values, 'go-', label='Train Val PSNR', 
                       linewidth=2, markersize=4)
    
    axes[0, 1].set_title('PSNR: Test vs Train Validation')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 总损失（训练）
    total_loss_steps, total_loss_values = extract_scalar_data(ea, 'train_loss_patches/total_loss')
    if total_loss_steps:
        axes[1, 0].plot(total_loss_steps, total_loss_values, 'r-', label='Training Total Loss', 
                       linewidth=1, alpha=0.8)
        axes[1, 0].set_title('Training Total Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Loss差异分析（如果有数据的话）
    if test_l1_steps and train_val_l1_steps:
        # 找到相同的迭代步数进行对比
        common_steps = list(set(test_l1_steps) & set(train_val_l1_steps))
        if common_steps:
            common_steps.sort()
            test_l1_dict = dict(zip(test_l1_steps, test_l1_values))
            train_l1_dict = dict(zip(train_val_l1_steps, train_val_l1_values))
            
            test_at_common = [test_l1_dict[step] for step in common_steps if step in test_l1_dict]
            train_at_common = [train_l1_dict[step] for step in common_steps if step in train_l1_dict]
            
            if len(test_at_common) == len(train_at_common) and test_at_common:
                diff = [abs(t - v) for t, v in zip(test_at_common, train_at_common)]
                axes[1, 1].plot(common_steps, diff, 'mo-', label='|Test - Train Val| L1 Loss', 
                               linewidth=2, markersize=3)
                axes[1, 1].set_title('Validation Gap Analysis')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Absolute Difference')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
    
    # 如果没有gap分析，显示迭代时间
    if not (test_l1_steps and train_val_l1_steps):
        iter_time_steps, iter_time_values = extract_scalar_data(ea, 'iter_time')
        if iter_time_steps:
            axes[1, 1].plot(iter_time_steps, iter_time_values, 'g-', linewidth=1, alpha=0.8)
            axes[1, 1].set_title('Iteration Time')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Time (ms)')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_vs_validation.png'), dpi=300, bbox_inches='tight')
    print(f"训练vs验证对比图已保存到: {os.path.join(output_dir, 'training_vs_validation.png')}")
    
    # 3. 绘制单独的Loss曲线（用于论文等）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取训练损失数据
    steps, l1_values = extract_scalar_data(ea, 'train_loss_patches/l1_loss')
    _, total_values = extract_scalar_data(ea, 'train_loss_patches/total_loss')
    
    if steps:
        ax.plot(steps, l1_values, 'b-', label='L1 Loss', linewidth=2, alpha=0.8)
        ax.plot(steps, total_values, 'r-', label='Total Loss', linewidth=2, alpha=0.8)
        
        ax.set_title('3D Gaussian Splatting Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加一些统计信息
        if l1_values:
            min_l1 = min(l1_values)
            final_l1 = l1_values[-1]
            ax.text(0.02, 0.98, f'Final L1 Loss: {final_l1:.6f}\nMin L1 Loss: {min_l1:.6f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    print(f"损失曲线图已保存到: {os.path.join(output_dir, 'loss_curve.png')}")

def print_summary_stats(ea):
    """打印训练统计信息"""
    print("\n=== 训练统计摘要 ===")
    
    # 训练损失统计
    steps, l1_values = extract_scalar_data(ea, 'train_loss_patches/l1_loss')
    if l1_values:
        print(f"训练 L1 Loss:")
        print(f"  初始值: {l1_values[0]:.6f}")
        print(f"  最终值: {l1_values[-1]:.6f}")
        print(f"  最小值: {min(l1_values):.6f}")
        print(f"  平均值: {np.mean(l1_values):.6f}")
        print(f"  总迭代数: {len(l1_values)}")
    
    # 总损失统计
    _, total_values = extract_scalar_data(ea, 'train_loss_patches/total_loss')
    if total_values:
        print(f"训练 Total Loss:")
        print(f"  初始值: {total_values[0]:.6f}")
        print(f"  最终值: {total_values[-1]:.6f}")
        print(f"  最小值: {min(total_values):.6f}")
        print(f"  平均值: {np.mean(total_values):.6f}")
    
    print("\n=== 验证统计摘要 ===")
    
    # 测试集L1 Loss统计
    test_steps, test_l1_values = extract_scalar_data(ea, 'test/loss_viewpoint - l1_loss')
    if test_l1_values:
        print(f"测试集 L1 Loss:")
        print(f"  最小值: {min(test_l1_values):.6f}")
        print(f"  最终值: {test_l1_values[-1]:.6f}")
        print(f"  平均值: {np.mean(test_l1_values):.6f}")
        print(f"  评估次数: {len(test_l1_values)}")
    
    # 训练验证集L1 Loss统计
    train_val_steps, train_val_l1_values = extract_scalar_data(ea, 'train/loss_viewpoint - l1_loss')
    if train_val_l1_values:
        print(f"训练验证集 L1 Loss:")
        print(f"  最小值: {min(train_val_l1_values):.6f}")
        print(f"  最终值: {train_val_l1_values[-1]:.6f}")
        print(f"  平均值: {np.mean(train_val_l1_values):.6f}")
        print(f"  评估次数: {len(train_val_l1_values)}")
    
    # PSNR统计
    test_psnr_steps, test_psnr = extract_scalar_data(ea, 'test/loss_viewpoint - psnr')
    if test_psnr:
        print(f"测试集 PSNR:")
        print(f"  最大值: {max(test_psnr):.2f} dB")
        print(f"  最终值: {test_psnr[-1]:.2f} dB")
        print(f"  平均值: {np.mean(test_psnr):.2f} dB")
    
    train_psnr_steps, train_psnr = extract_scalar_data(ea, 'train/loss_viewpoint - psnr')
    if train_psnr:
        print(f"训练验证集 PSNR:")
        print(f"  最大值: {max(train_psnr):.2f} dB")
        print(f"  最终值: {train_psnr[-1]:.2f} dB")
        print(f"  平均值: {np.mean(train_psnr):.2f} dB")
    
    # 过拟合分析
    if test_l1_values and train_val_l1_values:
        print(f"\n=== 过拟合分析 ===")
        final_test_l1 = test_l1_values[-1]
        final_train_l1 = train_val_l1_values[-1]
        gap = abs(final_test_l1 - final_train_l1)
        gap_percent = (gap / final_test_l1) * 100
        print(f"最终测试L1损失: {final_test_l1:.6f}")
        print(f"最终训练验证L1损失: {final_train_l1:.6f}")
        print(f"泛化间隙: {gap:.6f} ({gap_percent:.2f}%)")
        
        if gap_percent < 5:
            print("✓ 泛化良好，无明显过拟合")
        elif gap_percent < 15:
            print("⚠ 轻微过拟合")
        else:
            print("⚠ 可能存在过拟合")
    
    # 训练效率统计
    iter_time_steps, iter_times = extract_scalar_data(ea, 'iter_time')
    if iter_times:
        print(f"\n=== 训练效率统计 ===")
        print(f"平均迭代时间: {np.mean(iter_times):.2f} ms")
        print(f"最快迭代时间: {min(iter_times):.2f} ms")
        print(f"最慢迭代时间: {max(iter_times):.2f} ms")
        total_time_hours = (sum(iter_times) / 1000) / 3600
        print(f"总训练时间: {total_time_hours:.2f} 小时")
    
    # Gaussian点数统计
    point_steps, point_counts = extract_scalar_data(ea, 'total_points')
    if point_counts:
        print(f"\n=== Gaussian点数统计 ===")
        print(f"初始点数: {point_counts[0]:,}")
        print(f"最终点数: {point_counts[-1]:,}")
        print(f"最大点数: {max(point_counts):,}")
        print(f"平均点数: {np.mean(point_counts):,.0f}")
        growth_rate = ((point_counts[-1] - point_counts[0]) / point_counts[0]) * 100
        print(f"点数增长率: {growth_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='读取TensorBoard日志并绘制loss曲线')
    parser.add_argument('--log_dir', type=str, required=True, 
                       help='TensorBoard日志目录路径')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='输出图片保存目录 (默认: ./plots)')
    parser.add_argument('--show_stats', action='store_true',
                       help='显示训练统计信息')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"错误: 日志目录不存在: {args.log_dir}")
        return
    
    try:
        # 读取TensorBoard日志
        ea = read_tensorboard_logs(args.log_dir)
        
        # 绘制曲线
        plot_loss_curves(ea, args.output_dir)
        
        # 显示统计信息
        if args.show_stats:
            print_summary_stats(ea)
        
        print(f"\n所有图片已保存到目录: {args.output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()