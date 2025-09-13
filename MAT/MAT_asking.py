from generate_image import ImageGenerator

# 初始化生成器
generator = ImageGenerator(
    network_pkl='pretrained/CelebA-HQ_512.pkl',
    resolution=512,
    truncation_psi=1.0,
    noise_mode='const'
)

# 生成图像
image_paths = ['test_sets/CelebA-HQ/images/test1.png']
# mask_paths = ['test_sets/CelebA-HQ/masks/mask2.png']  # 可选
output_dir = 'samples'

generator.generate_images(
    image_paths=image_paths,
    mask_paths=mask_paths,
    output_dir=output_dir
)