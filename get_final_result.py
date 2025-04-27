import os
import json
import pandas as pd
import argparse
from collections import defaultdict

# 各类别对应的任务
CATEGORY_TASKS = {
    "Unstructured Text": ["natural_language_string"],
    "Structured Text": ["dict_search_string", "dict_search_number"],
    "Format": ["check_format", "output_format", "format_convert"],
    "Order": ["check_order", "character_order", "sentence_order", "word_order"],
    "Statistics": ["check_dedup", "character_dedup", "sentence_dedup", "word_dedup", "navigation_and_count", "relation_analysis"],
    "List Mapping": ["list_number"],
}


def read_jsonl(file_path):
    """读取单个jsonl文件"""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            results.append(data)
    return results

def compute_accuracy(jsonl_file):
    """计算单个jsonl文件准确率"""
    results = read_jsonl(jsonl_file)
    if len(results) == 0:
        return 0.0
    correct = sum(1 for item in results if item.get("is_right", False))
    return correct / len(results)

def aggregate_scores(input_folder):
    """收集每个模型的各类别得分"""
    model_scores = defaultdict(lambda: defaultdict(list))

    for task_folder in os.listdir(input_folder):
        task_folder_path = os.path.join(input_folder, task_folder)
        if not os.path.isdir(task_folder_path):
            continue

        for file_name in os.listdir(task_folder_path):
            if not file_name.endswith(".jsonl"):
                continue
            model_name = file_name.replace(".jsonl", "")
            file_path = os.path.join(task_folder_path, file_name)
            acc = compute_accuracy(file_path)

            # 判断属于哪个大类
            matched = False
            for category, tasks in CATEGORY_TASKS.items():
                if task_folder in tasks:
                    model_scores[model_name][category].append(acc)
                    matched = True
                    break
            if not matched:
                print(f"[Warning] Task {task_folder} 未匹配到任何大类，跳过。")

    return model_scores

def compute_final_scores(model_scores):
    """整理成最终输出的表格"""
    rows = []
    for model_name, category_scores in model_scores.items():
        row = {"Model": model_name}
        final_avg_list = []

        for cat in ["Unstructured Text", "Structured Text", "Format", "Order", "Statistics", "List Mapping"]:
            if cat in category_scores:
                score = sum(category_scores[cat]) / len(category_scores[cat])
                row[cat] = round(score, 3)
                final_avg_list.append(score)
            else:
                row[cat] = ""

        if final_avg_list:
            row["Average"] = round(sum(final_avg_list) / len(final_avg_list), 3)
        else:
            row["Average"] = ""

        rows.append(row)

    # 排序：Model 按字母升序
    rows = sorted(rows, key=lambda x: x["Model"].lower())

    df = pd.DataFrame(rows)

    # 控制列顺序
    df = df[["Model", "Unstructured Text", "Structured Text", "Format", "Order", "Statistics", "List Mapping", "Average"]]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the log folder')
    args = parser.parse_args()

    input_folder = args.input
    output_file = input_folder.rstrip("/").rstrip("\\") + "_final.xlsx"

    model_scores = aggregate_scores(input_folder)
    final_df = compute_final_scores(model_scores)

    final_df.to_excel(output_file, index=False)
    print(f"[INFO] 成功保存到 {output_file}")

if __name__ == "__main__":
    main()
