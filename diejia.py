import os
import cv2

# 定义函数，用于图像叠加处理
def blend_images(folder_white, folder_mat, output_folder, weight_white=0.5, weight_mat=0.5):
    # 支持的图像扩展名
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

    # 读取并排序两个文件夹中的图像文件
    files_white = sorted([f for f in os.listdir(folder_white) if f.lower().endswith(valid_exts)])
    files_mat = sorted([f for f in os.listdir(folder_mat) if f.lower().endswith(valid_exts)])

    # 确保两个文件夹图像数量一致
    if len(files_white) != len(files_mat):
        print("两个文件夹中的图像数量不一致！")
        return

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有图像进行处理
    for filename_white, filename_mat in zip(files_white, files_mat):
        path_white = os.path.join(folder_white, filename_white)
        path_mat = os.path.join(folder_mat, filename_mat)

        img_white = cv2.imread(path_white)
        img_mat = cv2.imread(path_mat)

        if img_white is None or img_mat is None:
            print(f"读取失败：{filename_white} 或 {filename_mat}")
            continue

        # 若图像大小不同，统一为white图像大小
        if img_white.shape != img_mat.shape:
            img_mat = cv2.resize(img_mat, (img_white.shape[1], img_white.shape[0]))

        # 图像叠加
        blended = cv2.addWeighted(img_white, weight_white, img_mat, weight_mat, 0)

        # 保存结果
        output_path = os.path.join(output_folder, filename_white)
        cv2.imwrite(output_path, blended)
        print(f"保存叠加图像: {output_path}")

    print("✅ 批量图像叠加完成。")

