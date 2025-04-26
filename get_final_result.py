import argparse
import os
import json
import pandas as pd
from collections import defaultdict

# --------------- Mapping配置 -----------------
# 每一类对应哪些子任务
CATEGORY_TASKS = {
    "Unstructured Text": ["natural_language_string"],
    "Structured Text": ["dict_search_string", "dict_search_number"],
    "Format Rules": ["check_format", "output_format", "format_convert"],
    "Order Rules": ["check_order", "character_order", "sentence_order", "word_order"],
    "Duplication Check": ["check_dedup"],
    "De-Duplication": ["character_dedup", "sentence_dedup", "word_dedup"],
    "Navigation": ["navigation_and_count"],
    "Relation Analysis": ["relation_analysis"],
    "List Mapping": ["list_number"],
}

# --------------- Order Rules权重 ----------------
# 特别注意加权：character_order和word_order是90%，sentence_order是60%
ORDER_RULES_WEIGHT = {
    "character_order": 0.9,
    "sentence_order": 0.6,
    "word_order": 0.9,
    "check_order": 1.0,
}

# --------------- 脚本主逻辑 -----------------
def process_logs(input_folder):
    # 保存每个模型每个任务的正确率
    model_task_correct = defaultdict(lambda: defaultdict(list))

    # 遍历每一个任务文件夹
    for task_name in os.listdir(input_folder):
        task_folder = os.path.join(input_folder, task_name)
        if not os.path.isdir(task_folder):
            continue
        
        for file_name in os.listdir(task_folder):
            if not file_name.endswith(".jsonl"):
                continue
            model_name = file_name.replace(".jsonl", "")

            file_path = os.path.join(task_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                total = len(lines)
                correct = 0
                for line in lines:
                    data = json.loads(line)
                    if data.get("is_right") is True:
                        correct += 1
                if total > 0:
                    acc = correct / total
                    model_task_correct[model_name][task_name].append(acc)

    # 汇总每个模型的每个大类得分
    results = []
    for model_name, task_scores in model_task_correct.items():
        record = {"Model": model_name}
        scores = []
        for category, tasks in CATEGORY_TASKS.items():
            if category == "Order Rules":
                # 特殊处理Order Rules加权
                weighted_sum = 0
                total_weight = 0
                for t in tasks:
                    weight = ORDER_RULES_WEIGHT.get(t, 1.0)
                    t_scores = task_scores.get(t, [])
                    if t_scores:
                        weighted_sum += sum(t_scores) * weight
                        total_weight += weight
                if total_weight > 0:
                    category_score = weighted_sum / total_weight
                else:
                    category_score = None
            else:
                # 其他类别直接平均
                category_scores = []
                for t in tasks:
                    category_scores.extend(task_scores.get(t, []))
                category_score = sum(category_scores) / len(category_scores) if category_scores else None

            record[category] = round(category_score, 4) if category_score is not None else None
            if category_score is not None:
                scores.append(category_score)
        
        # 最后再加一列：整体Average
        if scores:
            record["Average"] = round(sum(scores) / len(scores), 4)
        else:
            record["Average"] = None

        results.append(record)

    df = pd.DataFrame(results)
    return df

# --------------- 主函数 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to logs folder")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_path = input_folder.rstrip("/").rstrip("\\") + "_final.xlsx"

    df = process_logs(input_folder)
    df = df[["Model", "Unstructured Text", "Structured Text", "Format Rules", "Order Rules", "Duplication Check", "De-Duplication", "Navigation", "Relation Analysis", "List Mapping", "Average"]]
    df.to_excel(output_path, index=False)
    print(f"✅ Successfully saved final results to: {output_path}")

if __name__ == "__main__":
    main()
