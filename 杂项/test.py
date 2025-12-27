import os
import shutil
import sys


def copy_specific_folders(source_path, target_path, start_num=1, end_num=100):
    """
    复制指定编号范围内的文件夹从源路径到目标路径

    参数:
        source_path: 源文件夹路径
        target_path: 目标文件夹路径
        start_num: 起始编号
        end_num: 结束编号
    """

    # 检查源路径是否存在
    if not os.path.exists(source_path):
        print(f"错误：源路径 '{source_path}' 不存在")
        return False

    # 创建目标路径（如果不存在）
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"已创建目标路径: {target_path}")

    # 获取源路径下所有文件夹
    try:
        all_items = os.listdir(source_path)
    except PermissionError:
        print(f"错误：没有权限访问源路径 '{source_path}'")
        return False

    # 筛选出文件夹并过滤编号
    folders_to_copy = []
    for item in all_items:
        item_path = os.path.join(source_path, item)
        if os.path.isdir(item_path):
            try:
                # 尝试将文件夹名转换为数字
                folder_num = int(item)
                if start_num <= folder_num <= end_num:
                    folders_to_copy.append((item, folder_num))
            except ValueError:
                # 如果文件夹名不是数字，则跳过
                continue

    # 按数字排序
    folders_to_copy.sort(key=lambda x: x[1])

    if not folders_to_copy:
        print(f"在 '{source_path}' 中没有找到编号 {start_num}-{end_num} 的文件夹")
        return True

    print(f"找到 {len(folders_to_copy)} 个需要复制的文件夹:")
    for folder_name, num in folders_to_copy:
        print(f"  {num}: {folder_name}")

    # 复制文件夹
    success_count = 0
    for folder_name, folder_num in folders_to_copy:
        src_folder = os.path.join(source_path, folder_name)
        dst_folder = os.path.join(target_path, folder_name)

        try:
            # 如果目标文件夹已存在，则先删除
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
                print(f"已覆盖现有文件夹: {folder_name}")

            # 复制文件夹
            shutil.copytree(src_folder, dst_folder)
            print(f"成功复制: {folder_name}")
            success_count += 1

        except Exception as e:
            print(f"复制文件夹 {folder_name} 时出错: {e}")

    print(f"\n操作完成! 成功复制 {success_count}/{len(folders_to_copy)} 个文件夹")
    return True


if __name__ == "__main__":
    # 设置路径参数
    source_directory = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI/all"
    target_directory = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI/801-1000"

    # 执行复制操作
    print("开始复制文件夹...")
    print(f"源路径: {source_directory}")
    print(f"目标路径: {target_directory}")
    print("-" * 50)

    success = copy_specific_folders(source_directory, target_directory, 801, 900)

    if success:
        print("\n任务完成! 文件夹已复制到指定位置。")
    else:
        print("\n任务执行过程中遇到错误。")
        sys.exit(1)