import cv2
import numpy as np
import random


def generate_random_shape_with_percentage(image, max_size=100, num_shapes=5, target_percentage=35):
    """
    在图像上生成随机形状的白色破损蒙版，确保所有白色区域的总面积占整张图片面积的特定百分比。
    :param image: 输入图像
    :param max_size: 随机形状的最大尺寸
    :param num_shapes: 生成的随机形状数量（仅作为初始估计）
    :param target_percentage: 目标白色区域占图片面积的百分比
    :return: 应用随机白色破损蒙版后的图像
    """
    height, width = image.shape[:2]
    total_area = height * width
    target_white_area = int(total_area * target_percentage / 100)

    current_white_area = 0
    shapes = []

    # 创建一个掩码来跟踪白色区域
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_shapes):
        # 如果当前白色区域已经超过目标，停止生成新的形状
        if current_white_area >= target_white_area:
            break

        # 随机生成形状的中心点
        center_x = random.randint(max_size // 2, width - max_size // 2)
        center_y = random.randint(max_size // 2, height - max_size // 2)

        # 计算剩余需要的白色区域面积
        remaining_area = target_white_area - current_white_area

        # 随机生成形状的大小，但不超过剩余需要的面积
        max_shape_area = min(max_size * max_size, remaining_area)
        size = int(np.sqrt(max_shape_area))
        size = max(10, size)  # 确保大小至少为10

        # 随机生成形状的轮廓点
        num_points = random.randint(8, 15)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = []
        for angle in angles:
            # 添加一些小的随机扰动，使形状更加自然
            x = int(center_x + size * np.cos(angle) * (1 + random.uniform(-0.3, 0.3)))
            y = int(center_y + size * np.sin(angle) * (1 + random.uniform(-0.3, 0.3)))
            points.append((x, y))

        # 使用曲线拟合生成平滑的轮廓
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)

        # 创建一个单独的掩码来计算当前形状的面积
        shape_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(shape_mask, [hull], -1, 1, thickness=cv2.FILLED)
        shape_area = np.sum(shape_mask)

        # 如果当前形状的面积超过剩余需要的面积，调整大小
        if shape_area > remaining_area:
            scale = np.sqrt(remaining_area / shape_area)
            new_size = int(size * scale)
            points = []
            for angle in angles:
                x = int(center_x + new_size * np.cos(angle) * (1 + random.uniform(-0.3, 0.3)))
                y = int(center_y + new_size * np.sin(angle) * (1 + random.uniform(-0.3, 0.3)))
                points.append((x, y))
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            shape_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(shape_mask, [hull], -1, 1, thickness=cv2.FILLED)
            shape_area = np.sum(shape_mask)

        # 将当前形状的面积添加到掩码
        cv2.drawContours(mask, [hull], -1, 1, thickness=cv2.FILLED)
        current_white_area += shape_area

        # 将形状信息保存到列表中
        shapes.append(hull)

    # 创建一个与原图大小相同的白色破损蒙版
    white_mask = np.zeros_like(image)
    white_mask[mask == 1] = (255, 255, 255)

    # 应用白色破损蒙版到原图
    result = np.copy(image)
    result[mask == 1] = white_mask[mask == 1]

    return result
