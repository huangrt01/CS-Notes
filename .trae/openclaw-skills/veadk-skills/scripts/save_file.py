import argparse
import os


def save_file(file_path: str, content: str) -> str:
    """
    保存文件到指定路径

    参数:
    - file_path (str): 文件保存的路径。
    - content (str): 文件内容。

    返回:
    - str: 保存状态，"successfully saved" 表示成功保存。
    """

    # create dir if not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return "successfully saved"


def main():
    parser = argparse.ArgumentParser(description="Save content to a file")

    parser.add_argument("--path", type=str, required=True, help="Path to save the file")

    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Content to write into the file",
    )

    args = parser.parse_args()

    result = save_file(args.path, args.content)
    print(result)


if __name__ == "__main__":
    main()
