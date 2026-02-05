#!/usr/bin/env python3
"""为当前目录下以数字命名的文件夹建立末级子目录索引。

索引键：末级子目录名称（例如：524092420239CT）
索引值：该末级子目录的完整路径

用法示例：
  # 生成索引（默认保存为 leaf_index.json）
  python index_leaf_dirs.py

  # 指定根目录和输出文件
  python index_leaf_dirs.py --root . --output leaf_index.json

  # 查询某个末级子目录名称
  python index_leaf_dirs.py --query 524092420239CT
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


def iter_numeric_folders(root_dir: str) -> List[str]:
    """返回 root_dir 下名称为数字的一级目录绝对路径列表。"""
    results: List[str] = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and entry.isdigit():
            results.append(full_path)
    return results


def find_leaf_dirs(root_dir: str) -> List[str]:
    """递归遍历 root_dir 下所有末级目录（无子目录）。"""
    leaf_dirs: List[str] = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:
            leaf_dirs.append(dirpath)
    return leaf_dirs


def build_index(root_dir: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """构建末级目录索引。

    返回：
        index: 末级目录名 -> 完整路径
        duplicates: 末级目录名 -> 所有重复路径列表
    """
    index: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for numeric_root in iter_numeric_folders(root_dir):
        for leaf_dir in find_leaf_dirs(numeric_root):
            leaf_name = os.path.basename(leaf_dir)
            if leaf_name in index:
                duplicates.setdefault(leaf_name, [index[leaf_name]]).append(leaf_dir)
                continue
            index[leaf_name] = leaf_dir

    return index, duplicates


def save_index(output_path: str, index: Dict[str, str], duplicates: Dict[str, List[str]]) -> None:
    payload = {
        "index": index,
        "duplicates": duplicates,
        "count": len(index),
        "duplicate_count": len(duplicates),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("index", {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为数字命名目录建立末级子目录索引")
    p.add_argument("--root", "-r", default="/media/wmx/KINGIDISK/", help="搜索根目录（默认 /media/wmx/KINGIDISK/）")
    p.add_argument("--output", "-o", default="leaf_index.json", help="索引输出文件（默认 leaf_index.json）")
    p.add_argument("--query", "-q", default="", help="查询末级目录名（如 524092420239CT）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = os.path.abspath(args.root)

    if args.query:
        if not os.path.exists(args.output):
            raise FileNotFoundError(f"索引文件不存在：{args.output}")
        index = load_index(args.output)
        result = index.get(args.query)
        if result:
            print(result)
        else:
            print("")
        return

    index, duplicates = build_index(root_dir)
    save_index(args.output, index, duplicates)
    print(f"已生成索引：{args.output}")
    print(f"唯一末级目录数：{len(index)}")
    if duplicates:
        print(f"发现重复末级目录名：{len(duplicates)}（已写入索引中的 duplicates 字段）")


if __name__ == "__main__":
    main()
