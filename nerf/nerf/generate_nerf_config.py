import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

def analyze_dataset(data_dir: Path) -> Dict[str, Any]:
    """分析数据集信息"""
    info = {
        'has_transforms': False,
        'train_count': 0,
        'val_count': 0,
        'test_count': 0,
        'image_width': None,
        'image_height': None,
        'camera_angle_x': None,
        'has_white_background': True  # COLMAP数据通常需要白色背景
    }
    
    # 检查transforms文件
    transforms_files = ['transforms_train.json', 'transforms_val.json', 'transforms_test.json']
    for tf in transforms_files:
        tf_path = data_dir / tf
        if tf_path.exists():
            info['has_transforms'] = True
            with open(tf_path, 'r') as f:
                data = json.load(f)
                
            if tf == 'transforms_train.json':
                info['train_count'] = len(data.get('frames', []))
                info['camera_angle_x'] = data.get('camera_angle_x')
            elif tf == 'transforms_val.json':
                info['val_count'] = len(data.get('frames', []))
            elif tf == 'transforms_test.json':
                info['test_count'] = len(data.get('frames', []))
    
    # 检查图像尺寸（从第一张训练图像）
    train_dir = data_dir / 'train'
    if train_dir.exists():
        image_files = list(train_dir.glob('*.png')) + list(train_dir.glob('*.jpg'))
        if image_files:
            try:
                import cv2
                img = cv2.imread(str(image_files[0]))
                if img is not None:
                    info['image_height'], info['image_width'] = img.shape[:2]
            except ImportError:
                print("警告: 无法导入cv2，无法检测图像尺寸")
    
    return info

def generate_config(data_dir: Path, exp_name: str = None, 
                   output_file: Path = None, **kwargs) -> str:
    """生成NeRF配置文件内容"""
    
    # 分析数据集
    info = analyze_dataset(data_dir)
    
    if not info['has_transforms']:
        raise ValueError(f"在 {data_dir} 中未找到transforms文件，请确保已正确转换COLMAP数据")
    
    # 设置默认实验名
    if exp_name is None:
        exp_name = f"colmap_{data_dir.name}"
    
    # 基础配置
    config_lines = [
        f"expname = {exp_name}",
        f"basedir = ./logs",
        f"datadir = {data_dir.absolute()}",
        f"dataset_type = blender",
        "",
        "# 数据集信息",
        f"# 训练集: {info['train_count']} 张",
        f"# 验证集: {info['val_count']} 张", 
        f"# 测试集: {info['test_count']} 张",
    ]
    
    if info['image_width'] and info['image_height']:
        config_lines.extend([
            f"# 图像尺寸: {info['image_width']}x{info['image_height']}",
        ])
    
    config_lines.extend([
        "",
        "# 基础设置",
        "no_batching = True",
        "use_viewdirs = True",
    ])
    
    # 背景设置
    if info['has_white_background']:
        config_lines.append("white_bkgd = True")
    else:
        config_lines.append("white_bkgd = False")
    
    config_lines.extend([
        "",
        "# 学习率设置",
        "lrate_decay = 500",
        "",
        "# 采样设置", 
        "N_samples = 64",
        "N_importance = 128",
        "N_rand = 1024",
    ])
    
    # 根据图像数量调整采样设置
    total_images = info['train_count'] + info['val_count'] + info['test_count']
    if total_images < 100:
        # 图像较少时，增加采样点
        config_lines.extend([
            "",
            "# 图像较少，增加采样密度",
            "N_samples = 128", 
            "N_importance = 256",
        ])
    elif total_images > 500:
        # 图像较多时，可以适当减少采样
        config_lines.extend([
            "",
            "# 图像较多，可适当减少采样",
            "N_samples = 64",
            "N_importance = 128", 
        ])
    
    config_lines.extend([
        "",
        "# 预处理设置",
        "precrop_iters = 500",
        "precrop_frac = 0.5",
    ])
    
    # 根据图像尺寸决定是否使用half_res
    if info['image_width'] and info['image_height']:
        if info['image_width'] * info['image_height'] > 1920 * 1080:
            config_lines.append("half_res = True  # 大尺寸图像，使用半分辨率")
        else:
            config_lines.append("half_res = False  # 中等尺寸图像，使用全分辨率")
    else:
        config_lines.append("half_res = True  # 未知图像尺寸，建议使用半分辨率")
    
    # 添加用户自定义参数
    if kwargs:
        config_lines.extend([
            "",
            "# 自定义参数"
        ])
        for key, value in kwargs.items():
            if isinstance(value, bool):
                config_lines.append(f"{key} = {str(value)}")
            elif isinstance(value, (int, float)):
                config_lines.append(f"{key} = {value}")
            else:
                config_lines.append(f"{key} = {value}")
    
    config_content = '\n'.join(config_lines)
    
    # 保存到文件
    if output_file is None:
        output_file = data_dir / f"{exp_name}_config.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return config_content, output_file

def main():
    parser = argparse.ArgumentParser(description="为COLMAP转换的NeRF数据生成配置文件")
    parser.add_argument("data_dir", help="包含transforms文件的数据目录")
    parser.add_argument("--exp_name", help="实验名称，默认使用目录名")
    parser.add_argument("--output", help="输出配置文件路径")
    parser.add_argument("--no_white_bkgd", action="store_true", help="不使用白色背景")
    parser.add_argument("--full_res", action="store_true", help="强制使用全分辨率")
    parser.add_argument("--half_res", action="store_true", help="强制使用半分辨率")
    parser.add_argument("--N_samples", type=int, help="粗采样点数")
    parser.add_argument("--N_importance", type=int, help="细采样点数")
    parser.add_argument("--N_rand", type=int, help="每批随机光线数")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 检查必要文件
    required_files = ['transforms_train.json']
    missing_files = []
    for rf in required_files:
        if not (data_dir / rf).exists():
            missing_files.append(rf)
    
    if missing_files:
        print(f"错误: 缺少必要文件: {missing_files}")
        print("请确保已使用COLMAP转换脚本正确生成了NeRF格式数据")
        return
    
    # 设置输出文件
    output_file = None
    if args.output:
        output_file = Path(args.output)
    
    # 准备自定义参数
    custom_params = {}
    
    if args.no_white_bkgd:
        custom_params['white_bkgd'] = False
    
    if args.full_res:
        custom_params['half_res'] = False
    elif args.half_res:
        custom_params['half_res'] = True
        
    if args.N_samples:
        custom_params['N_samples'] = args.N_samples
    if args.N_importance:
        custom_params['N_importance'] = args.N_importance  
    if args.N_rand:
        custom_params['N_rand'] = args.N_rand
    
    try:
        config_content, output_path = generate_config(
            data_dir, 
            exp_name=args.exp_name,
            output_file=output_file,
            **custom_params
        )
        
        print("✅ 配置文件生成成功！")
        print(f"📁 保存位置: {output_path}")
        print(f"\n📋 配置文件内容:")
        print("-" * 50)
        print(config_content)
        print("-" * 50)
        print(f"\n🚀 使用方法:")
        print(f"python train.py --config {output_path}")
        
    except Exception as e:
        print(f"❌ 生成配置文件失败: {e}")

if __name__ == "__main__":
    main()