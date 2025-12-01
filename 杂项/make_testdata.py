def main():
    # 1. 定义目标全集：1 到 1000
    # range(1, 1001) 生成从 1 到 1000 的整数
    full_set = set(range(1, 1000))

    existing_numbers = set()

    # 2. 读取 ImageCAS.txt 文件内容
    try:
        with open('ImageCAS.txt', 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过空行，将其转换为整数存入集合
                if line:
                    existing_numbers.add(int(line))
    except FileNotFoundError:
        print("错误：找不到文件 ImageCAS.txt，请确保文件在同一目录下。")
        return

    # 3. 找出缺失的数字 (全集 减去 现有集合)
    # 使用集合差集操作，然后转回列表
    missing_numbers = list(full_set - existing_numbers)

    # 4. 按升序排列
    missing_numbers.sort()

    # 5. 直接 Print 出来
    print("在 1-1000 范围内缺失的数字（即测试集）：")
    print(missing_numbers)

    # 额外输出一下数量确认是否为100个
    print(f"\n共找到 {len(missing_numbers)} 个缺失数字。")


if __name__ == "__main__":
    main()