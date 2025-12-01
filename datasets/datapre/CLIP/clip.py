import os
import numpy as np
import nibabel as nib
from pathlib import Path


class NiftiCropper:
    def __init__(self, margin=(5, 5, 5), output_dir=None):
        self.margin = margin
        self.output_dir = output_dir

    def find_bounding_box(self, mask_data):
        """找到包含标签的最小边界框"""
        nonzero_indices = np.where(mask_data > 0)

        if len(nonzero_indices[0]) == 0:
            raise ValueError("标签图像中没有找到任何非零像素")

        min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
        min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
        min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def calculate_crop_indices(self, bbox, image_shape):
        """计算裁剪索引"""
        min_x, max_x, min_y, max_y, min_z, max_z = bbox

        crop_min_x = max(0, min_x - self.margin[0])
        crop_max_x = min(image_shape[0], max_x + self.margin[0] + 1)
        crop_min_y = max(0, min_y - self.margin[1])
        crop_max_y = min(image_shape[1], max_y + self.margin[1] + 1)
        crop_min_z = max(0, min_z - self.margin[2])
        crop_max_z = min(image_shape[2], max_z + self.margin[2] + 1)

        return (crop_min_x, crop_max_x, crop_min_y, crop_max_y, crop_min_z, crop_max_z)

    def update_affine_matrix(self, original_affine, crop_min_x, crop_min_y, crop_min_z):
        """更新仿射矩阵"""
        voxel_sizes = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))
        new_affine = original_affine.copy()
        new_affine[:3, 3] = original_affine[:3, 3] + [crop_min_x * voxel_sizes[0],
                                                      crop_min_y * voxel_sizes[1],
                                                      crop_min_z * voxel_sizes[2]]
        return new_affine

    def crop_image_pair(self, image_path, mask_path):
        """裁剪图像和标签对"""
        # 加载数据
        image_img = nib.load(image_path)
        mask_img = nib.load(mask_path)
        image_data = image_img.get_fdata()
        mask_data = mask_img.get_fdata()

        # 找到边界框并计算裁剪范围
        bbox = self.find_bounding_box(mask_data)
        crop_indices = self.calculate_crop_indices(bbox, image_data.shape)
        crop_min_x, crop_max_x, crop_min_y, crop_max_y, crop_min_z, crop_max_z = crop_indices

        # 执行裁剪
        cropped_image = image_data[crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z]
        cropped_mask = mask_data[crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z]

        # 更新仿射矩阵
        new_affine = self.update_affine_matrix(image_img.affine, crop_min_x, crop_min_y, crop_min_z)

        # 创建新图像
        cropped_image_img = nib.Nifti1Image(cropped_image, new_affine)
        cropped_mask_img = nib.Nifti1Image(cropped_mask, new_affine)

        # 保存结果 - 修正：同时保存图像和标签
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            image_filename = Path(image_path).name
            mask_filename = Path(mask_path).name
            image_output_path = os.path.join(self.output_dir, "img_"+image_filename)
            mask_output_path = os.path.join(self.output_dir, "label_"+mask_filename)


        # 保存图像和标签
        nib.save(cropped_image_img, image_output_path)
        nib.save(cropped_mask_img, mask_output_path)  # 添加了这行代码

        print(f"裁剪完成: {Path(image_path).name} -> 尺寸 {image_data.shape} -> {cropped_image.shape}")
        print(f"标签保存: {Path(mask_path).name} -> 尺寸 {mask_data.shape} -> {cropped_mask.shape}")
        return image_output_path, mask_output_path

    def process_folder(self, image_dir, mask_dir):
        """处理整个文件夹"""
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

        for img_file in image_files:
            # 简单匹配：将.img替换为.label
            mask_file = img_file.replace('.img.', '.label.')
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                print(f"未找到对应的标签文件: {mask_file}")
                continue

            image_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                self.crop_image_pair(image_path, mask_path)
            except Exception as e:
                print(f"处理 {img_file} 时出错: {e}")


if __name__ == "__main__":
    # 使用示例
    image_dir = "/home/yangrui/Project/Base-models/input/Parse/img"
    mask_dir = "/home/yangrui/Project/Base-models/input/Parse/label"
    output_dir = "/home/yangrui/Project/Base-models/input/Parse/CLIP"

    cropper = NiftiCropper(margin=(5, 5, 5), output_dir=output_dir)
    cropper.process_folder(image_dir, mask_dir)