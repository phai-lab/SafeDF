import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import math
import time
from torchvision import transforms
import random

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from efficientvit.models.utils import resize
from applications.efficientvit_seg.eval_efficientvit_seg_model import ADE20KDataset, CityscapesDataset, Resize, ToTensor, get_canvas

def test_single_image():
    """严格按照官方demo的数据预处理和推理流程"""
    
    # 模型配置
    model_name = "efficientvit-seg-l2-cityscapes"
    weight_path = "./efficientvit_seg_l2_cityscapes.pt"

    model_name = "efficientvit-seg-l2-ade20k"
    weight_path = "./l2.pt"
    
    # # 三个测试图像路径 #4192
    # test_images = [ #4192
    #     # "/home/nuo/OpenFly-Platform/extracted_trajectory_data_v4/20250106_133411.png", #4192
    #     # "/home/nuo/OpenFly-Platform/extracted_trajectory_data_v4_real/20250114_065933_10.png", #4192
    #     "/home/nuo/MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031452.791720.png" #4192
    # ] #4192

    image_dir = "/home/nuo/MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_desk/rgb"
    image_files = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print(f"错误：在目录中找不到图像: {image_dir}")
        return
    num_runs = 100
    
    # 检查权重文件是否存在
    if not os.path.exists(weight_path):
        print(f"错误：权重文件不存在: {weight_path}")
        return
    
    print("=== EfficientViT分割模型测试（严格按照官方demo） ===")
    print(f"模型: {model_name}")
    print(f"权重: {weight_path}")
    
    # 加载模型
    print("正在加载模型...")
    model = create_efficientvit_seg_model(model_name, weight_url=weight_path).cuda()
    model.eval()
    inference_times_all = []
    last_image = None
    last_output = None
    last_image_path = None
    
    # # 遍历三个测试图像 #4192
    # for i, test_image_path in enumerate(test_images, 1): #4192
    #     print(f"\n{'='*60}") #4192
    #     print(f"测试图像 {i}/3: {os.path.basename(test_image_path)}") #4192
    #     print(f"{'='*60}") #4192
    #      #4192
    #     # 检查图像文件是否存在 #4192
    #     if not os.path.exists(test_image_path): #4192
    #         print(f"错误：图像文件不存在: {test_image_path}") #4192
    #         continue #4192
    #      #4192
    #     # 严格按照官方demo的预处理流程 #4192
    #     print("正在加载图像...") #4192
    #     image = np.array(Image.open(test_image_path).convert("RGB")) #4192
    #     print(f"原始图像尺寸: {image.shape}") #4192
    #  #4192
    #     # downsample 2 #4192
    #     downsample_factor = 4 #4192
    #     # raise NotImplementedError(image.shape[0]) #4192
    #     image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_LINEAR) #4192
    #     image = cv2.resize(image, (image.shape[1]//downsample_factor, image.shape[0]//downsample_factor), interpolation=cv2.INTER_LINEAR) #4192
    #  #4192
    #     # 数据预处理 - 严格按照官方demo #4192
    #     data = image #4192
    #     dataset = "cityscapes" #4192
    #     dataset = "ade20k" #4192
    #     crop_size = 1024  # 使用更大的crop_size #4192
    #      #4192
    #     if dataset == "cityscapes": #4192
    #         transform = transforms.Compose( #4192
    #             [ #4192
    #                 Resize((crop_size, crop_size * 2)), #4192
    #                 ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #4192
    #             ] #4192
    #         ) #4192
    #         class_colors = CityscapesDataset.class_colors #4192
    #     else: #4192
    #         transform = transforms.Compose( #4192
    #             [ #4192
    #                 ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #4192
    #             ] #4192
    #         ) #4192
    #         class_colors = ADE20KDataset.class_colors #4192
    #  #4192
    #         # raise NotImplementedError #4192
    #      #4192
    #     # 严格按照官方demo的transform调用方式 #4192
    #     data = transform({"data": data, "label": np.ones_like(data)})["data"] #4192
    #      #4192
    #     # 推理 - 严格按照官方demo #4192
    #     print("正在进行推理...") #4192
    #     os.makedirs("./efficientvit_results", exist_ok=True) #4192
    #      #4192
    #     # 预热GPU #4192
    #     with torch.inference_mode(): #4192
    #         _ = model(torch.unsqueeze(data, dim=0).cuda()) #4192
    #      #4192
    #     # 测试推理时间 #4192
    #     num_runs = 100 #4192
    #     inference_times = [] #4192
    #      #4192
    #     with torch.inference_mode(): #4192
    #         for _ in range(num_runs): #4192
    #             start_time = time.time() #4192
    #              #4192
    #             data_tensor = torch.unsqueeze(data, dim=0).cuda() #4192
    #             output = model(data_tensor) #4192
    #              #4192
    #             # resize the output to match the shape of the mask #4192
    #             if output.shape[-2:] != image.shape[:2]: #4192
    #                 output = resize(output, size=image.shape[:2]) #4192
    #              #4192
    #             output = torch.argmax(output, dim=1).cpu().numpy()[0] #4192
    #              #4192
    #             end_time = time.time() #4192
    #             inference_times.append(end_time - start_time) #4192
    #      #4192
    #     # 计算平均推理时间和FPS #4192
    #     avg_inference_time = np.mean(inference_times) #4192
    #     fps = 1.0 / avg_inference_time #4192
    #     std_inference_time = np.std(inference_times) #4192
    #      #4192
    #     print(f"推理性能统计 ({num_runs}次平均):") #4192
    #     print(f"  平均推理时间: {avg_inference_time:.4f}秒 ± {std_inference_time:.4f}秒") #4192
    #     print(f"  平均FPS: {fps:.2f}") #4192
    #     print(f"  最快推理时间: {min(inference_times):.4f}秒") #4192
    #     print(f"  最慢推理时间: {max(inference_times):.4f}秒") #4192
    #      #4192
    #     # 使用最后一次推理结果进行可视化 #4192
    #     with torch.inference_mode(): #4192
    #         data_tensor = torch.unsqueeze(data, dim=0).cuda() #4192
    #         output = model(data_tensor) #4192
    #          #4192
    #         # resize the output to match the shape of the mask #4192
    #         if output.shape[-2:] != image.shape[:2]: #4192
    #             output = resize(output, size=image.shape[:2]) #4192
    #          #4192
    #         output = torch.argmax(output, dim=1).cpu().numpy()[0] #4192
    #          #4192
    #         # 使用官方demo的可视化函数 - 纯分割颜色 #4192
    #         canvas = get_canvas(image, output, class_colors, opacity=1.0) #4192
    #          #4192
    #         # 保存结果 #4192
    #         base_name = os.path.splitext(os.path.basename(test_image_path))[0] #4192
    #         output_path = f"./efficientvit_results/{base_name}_demo.png" #4192
    #         Image.fromarray(canvas).save(output_path) #4192
    #          #4192
    #         print(f"结果保存到: {output_path}") #4192
    #          #4192
    #         # 同时保存混合版本用于对比 #4192
    #         canvas_mixed = get_canvas(image, output, class_colors, opacity=0.5) #4192
    #         output_path_mixed = f"./efficientvit_results/{base_name}_demo_mixed.png" #4192
    #         Image.fromarray(canvas_mixed).save(output_path_mixed) #4192
    #         print(f"混合结果保存到: {output_path_mixed}") #4192
    #      #4192
    #     # 统计信息 #4192
    #     unique_classes, counts = np.unique(output, return_counts=True) #4192
    #     print(f"检测到的类别: {len(unique_classes)}") #4192
    #     for class_id, count in zip(unique_classes, counts): #4192
    #         if class_id < len(CityscapesDataset.classes): #4192
    #             class_name = CityscapesDataset.classes[class_id] #4192
    #             class_percentage = count / output.size * 100 #4192
    #             print(f"  {class_name}: {count:,} 像素 ({class_percentage:.1f}%)") #4192
    #         else: #4192
    #             print(f"  未知类别{class_id}: {count:,} 像素 ({count/output.size*100:.1f}%)") #4192

    dataset_name = "ade20k"
    crop_size = 1024
    if dataset_name == "cityscapes":
        transform = transforms.Compose(
            [
                Resize((crop_size, crop_size * 2)),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = CityscapesDataset.class_colors
    else:
        transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = ADE20KDataset.class_colors

    os.makedirs("./efficientvit_results", exist_ok=True)

    for i in range(1, num_runs + 1):
        test_image_path = random.choice(image_files)
        print(f"\n{'='*60}")
        print(f"测试图像 {i}/{num_runs}: {os.path.basename(test_image_path)}")
        print(f"{'='*60}")

        if not os.path.exists(test_image_path):
            print(f"错误：图像文件不存在: {test_image_path}")
            continue

        print("正在加载图像...")
        image_full = np.array(Image.open(test_image_path).convert("RGB"))
        print(f"原始图像尺寸: {image_full.shape}")

        downsample_factor = 1
        image = cv2.resize(image_full, (512, 384), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(
            image,
            (
                max(1, image.shape[1] // downsample_factor),
                max(1, image.shape[0] // downsample_factor),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        data = transform({"data": image, "label": np.ones_like(image)})["data"]

        with torch.inference_mode():
            _ = model(torch.unsqueeze(data, dim=0).cuda())

        start_time = time.time()
        with torch.inference_mode():
            data_tensor = torch.unsqueeze(data, dim=0).cuda()
            output = model(data_tensor)
            if output.shape[-2:] != image.shape[:2]:
                output = resize(output, size=image.shape[:2])
            output_np = torch.argmax(output, dim=1).cpu().numpy()[0]
        end_time = time.time()

        inference_time = end_time - start_time
        inference_times_all.append(inference_time)

        avg_inference_time = inference_time
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else float("inf")
        print("推理性能统计 (单次):")
        print(f"  推理时间: {avg_inference_time:.4f}秒")
        print(f"  FPS: {fps:.2f}")

        last_image = image
        last_output = output_np
        last_image_path = test_image_path

        canvas = get_canvas(last_image, last_output, class_colors, opacity=1.0)
        base_name = os.path.splitext(os.path.basename(last_image_path))[0]
        output_path = f"./efficientvit_results/{base_name}_demo.png"
        Image.fromarray(canvas).save(output_path)
        print(f"结果保存到: {output_path}")

        canvas_mixed = get_canvas(last_image, last_output, class_colors, opacity=0.5)
        output_path_mixed = f"./efficientvit_results/{base_name}_demo_mixed.png"
        Image.fromarray(canvas_mixed).save(output_path_mixed)
        print(f"混合结果保存到: {output_path_mixed}")

        # print resolution
        print(f"当前图像分辨率: {image.shape[1]}x{image.shape[0]}")

    
    if inference_times_all:
        avg_inference_time = np.mean(inference_times_all)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else float("inf")
        std_inference_time = np.std(inference_times_all)
        print("\n整体推理性能统计:")
        print(f"  平均推理时间: {avg_inference_time:.4f}秒 ± {std_inference_time:.4f}秒")
        print(f"  平均FPS: {fps:.2f}")
        print(f"  最快推理时间: {min(inference_times_all):.4f}秒")
        print(f"  最慢推理时间: {max(inference_times_all):.4f}秒")

    if last_output is not None and last_image is not None and last_image_path is not None:
        canvas = get_canvas(last_image, last_output, class_colors, opacity=1.0)
        base_name = os.path.splitext(os.path.basename(last_image_path))[0]
        output_path = f"./efficientvit_results/{base_name}_demo.png"
        Image.fromarray(canvas).save(output_path)
        print(f"结果保存到: {output_path}")

        canvas_mixed = get_canvas(last_image, last_output, class_colors, opacity=0.5)
        output_path_mixed = f"./efficientvit_results/{base_name}_demo_mixed.png"
        Image.fromarray(canvas_mixed).save(output_path_mixed)
        print(f"混合结果保存到: {output_path_mixed}")

        unique_classes, counts = np.unique(last_output, return_counts=True)
        print(f"检测到的类别: {len(unique_classes)}")
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(ADE20KDataset.classes):
                class_name = ADE20KDataset.classes[class_id]
                class_percentage = count / last_output.size * 100
                print(f"  {class_name}: {count:,} 像素 ({class_percentage:.1f}%)")
            else:
                class_percentage = count / last_output.size * 100
                print(f"  未知类别{class_id}: {count:,} 像素 ({class_percentage:.1f}%)")

    print(f"\n{'='*60}")
    print("=== 所有测试完成 ===")
    print(f"结果保存在: ./efficientvit_results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_single_image() 
