import SimpleITK as sitk
from pathlib import Path

# ================= 配置区域 =================
# 你的数据根目录
ROOT_DIR = Path(
    "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-resize/val")

# Z轴切片数的阈值
Z_AXIS_THRESHOLD = 300


# ===========================================

def check_image_sizes():
    if not ROOT_DIR.exists():
        print(f"❌ 错误：目录不存在 {ROOT_DIR}")
        return

    print(f"📂 正在扫描目录: {ROOT_DIR}")
    print("-" * 60)
    print(f"{'Case ID':<10} | {'尺寸 (W, H, D)':<25} | {'Z轴切片数':<12} | {'状态'}")
    print("-" * 60)

    # 获取所有子文件夹，并尝试按数字排序 (1, 2, 3...)
    subdirs = [d for d in ROOT_DIR.iterdir() if d.is_dir()]
    # 排序逻辑：如果是数字就按数字排，否则按字符串排
    subdirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else x.name)

    count = 0
    z_axis_over_50 = []  # 存储Z轴切片数超过50的案例

    for folder in subdirs:
        case_id = folder.name
        # 根据你的命名规则：文件夹名 1 -> 图像名 1.img.nii.gz
        img_path = folder / f"{case_id}.img.nii.gz"

        if not img_path.exists():
            # 尝试模糊搜索，防止命名不一致
            potential_files = list(folder.glob("*.img.nii.gz"))
            if potential_files:
                img_path = potential_files[0]
            else:
                print(f"{case_id:<10} | {'-':<25} | {'-':<12} | ❌ 文件缺失")
                continue

        try:
            # 高速读取模式：只读头信息，不读像素数据
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(img_path))
            reader.ReadImageInformation()

            size = reader.GetSize()  # 返回 (Width, Height, Depth)
            z_slices = size[2]  # 获取Z轴切片数（Depth）

            status = "✅ 正常"
            if z_slices > Z_AXIS_THRESHOLD:
                z_axis_over_50.append((case_id, z_slices))
                status = "🔍 Z>50"

            print(f"{case_id:<10} | {str(size):<25} | {z_slices:<12} | {status}")
            count += 1

        except Exception as e:
            print(f"{case_id:<10} | {'Error':<25} | {'-':<12} | ❌ 读取失败: {e}")

    print("-" * 60)
    print(f"统计完成，共检测 {count} 个图像。")

    # 打印Z轴切片数超过50的案例统计
    print("\n" + "=" * 50)
    print("📊 Z轴切片数超过50的案例统计")
    print("=" * 50)

    if z_axis_over_50:
        print(f"找到 {len(z_axis_over_50)} 个Z轴切片数超过50的案例：")
        for case_id, z_slices in z_axis_over_50:
            print(f"  • 案例 {case_id}: {z_slices} 个切片")

        # 计算统计信息
        max_z = max(z_slices for _, z_slices in z_axis_over_50)
        min_z = min(z_slices for _, z_slices in z_axis_over_50)
        avg_z = sum(z_slices for _, z_slices in z_axis_over_50) / len(z_axis_over_50)

        print(f"\n📈 统计信息：")
        print(f"  最多切片数: {max_z}")
        print(f"  最少切片数: {min_z}")
        print(f"  平均切片数: {avg_z:.1f}")
        print(f"  占比: {len(z_axis_over_50) / count * 100:.1f}% ({len(z_axis_over_50)}/{count})")
    else:
        print("❌ 未找到Z轴切片数超过50的案例")


if __name__ == "__main__":
    check_image_sizes()