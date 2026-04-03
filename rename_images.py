import os
import re


def rename_images(folder_path: str) -> None:
    """
    将指定文件夹下命名格式为 '0xxxX2.png' 的图片重命名为 '0xxx.png'

    :param folder_path: 图片所在文件夹路径
    """
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹不存在 -> {folder_path}")
        return

    # 匹配 0xxxX8.png 格式，xxx 为任意字符
    pattern = re.compile(r'^(0.+)X8(\.png)$', re.IGNORECASE)

    renamed_count = 0
    skipped_count = 0

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            new_name = match.group(1) + match.group(2)
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            if os.path.exists(new_path):
                print(f"跳过（目标文件已存在）：{filename} -> {new_name}")
                skipped_count += 1
                continue

            os.rename(old_path, new_path)
            print(f"已重命名：{filename} -> {new_name}")
            renamed_count += 1

    print(f"\n完成！共重命名 {renamed_count} 个文件，跳过 {skipped_count} 个文件。")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python rename_images.py <文件夹路径>")
        print("示例：python rename_images.py ./images")
        sys.exit(1)

    target_folder = sys.argv[1]
    rename_images(target_folder)
