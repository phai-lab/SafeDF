import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import math
from torchvision import transforms

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from efficientvit.models.utils import resize
from applications.efficientvit_seg.eval_efficientvit_seg_model import ADE20KDataset, CityscapesDataset, Resize, ToTensor, get_canvas

def debug_mapping():
    """调试类别映射和颜色分配"""
    
    # 模型配置
    model_name = "efficientvit-seg-l2-cityscapes"
    weight_path = "./efficientvit_seg_l2_cityscapes.pt"
    test_image_path = "/home/nuo/OpenFly-Platform/extracted_trajectory_data_v4_real/20250114_065933_10.png"
    
    print("=== 调试类别映射 ===")
    
    # 检查类别和颜色定义
    print("Cityscapes类别数量:", len(CityscapesDataset.classes))
    print("Cityscapes颜色数量:", len(CityscapesDataset.class_colors))
    
    print("\n类别列表:")
    for i, (class_name, color) in enumerate(zip(CityscapesDataset.classes, CityscapesDataset.class_colors)):
        print(f"  ID {i}: {class_name} -> RGB{color}")
    
    # 加载模型和推理
    print("\n正在加载模型...")
    model = create_efficientvit_seg_model(model_name, weight_url=weight_path).cuda()
    model.eval()
    
    # 加载图像
    print("正在加载图像...")
    image = np.array(Image.open(test_image_path).convert("RGB"))
    print(f"原始图像尺寸: {image.shape}")
    
    # 数据预处理
    data = image
    crop_size = 1024
    
    transform = transforms.Compose([
        Resize((crop_size, crop_size * 2)),
        ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data = transform({"data": data, "label": np.ones_like(data)})["data"]
    
    # 推理
    print("正在进行推理...")
    with torch.inference_mode():
        data = torch.unsqueeze(data, dim=0).cuda()
        output = model(data)
        
        if output.shape[-2:] != image.shape[:2]:
            output = resize(output, size=image.shape[:2])
        
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # 分析预测结果
    print(f"\n预测结果分析:")
    print(f"预测形状: {pred.shape}")
    print(f"预测值范围: {pred.min()} ~ {pred.max()}")
    
    unique_classes, counts = np.unique(pred, return_counts=True)
    print(f"检测到的类别ID: {unique_classes}")
    print(f"对应的像素数量: {counts}")
    
    print(f"\n详细类别分布:")
    for class_id, count in zip(unique_classes, counts):
        if class_id < len(CityscapesDataset.classes):
            class_name = CityscapesDataset.classes[class_id]
            color = CityscapesDataset.class_colors[class_id]
            percentage = count / pred.size * 100
            print(f"  ID {class_id}: {class_name} -> RGB{color} -> {count:,} 像素 ({percentage:.1f}%)")
        else:
            print(f"  ID {class_id}: 超出范围 -> {count:,} 像素")
    
    # 检查缺失的类别
    missing_classes = set(range(len(CityscapesDataset.classes))) - set(unique_classes)
    if missing_classes:
        print(f"\n缺失的类别ID: {missing_classes}")
        for missing_id in missing_classes:
            class_name = CityscapesDataset.classes[missing_id]
            color = CityscapesDataset.class_colors[missing_id]
            print(f"  ID {missing_id}: {class_name} -> RGB{color} (未检测到)")
    
    # 测试get_canvas函数
    print(f"\n测试get_canvas函数...")
    canvas = get_canvas(image, pred, CityscapesDataset.class_colors)
    
    # 保存结果
    os.makedirs("./efficientvit_results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(test_image_path))[0]
    output_path = f"./efficientvit_results/{base_name}_debug.png"
    Image.fromarray(canvas).save(output_path)
    print(f"调试结果保存到: {output_path}")
    
    # 检查canvas的颜色分布
    canvas_unique = np.unique(canvas.reshape(-1, 3), axis=0)
    print(f"\nCanvas中的唯一颜色数量: {len(canvas_unique)}")
    print(f"Canvas颜色范围: RGB{canvas.min(axis=(0,1))} ~ RGB{canvas.max(axis=(0,1))}")

if __name__ == "__main__":
    debug_mapping() 