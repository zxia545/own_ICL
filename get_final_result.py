import pandas as pd
import argparse
import os

# ------------------------------------------------------
# 配置字段分类
# ------------------------------------------------------

unstructured_fields = [
    'natural_language_string_natural_language_hash_string_copying'
]

structured_fields = [
    'dict_search_number_dict_search_number-all_similar',
    'dict_search_number_dict_search_number-half_similar',
    'dict_search_number_dict_search_number-non_similar',
    'dict_search_string_dict_search_hash_string',
    'list_number_list_number_list_number'
]

format_fields = [
    'check_format_format_convert_normal',
    'check_format_format_convert_transfer',
    'output_format_output_format_output_format_01',
    'output_format_output_format_output_format_02',
    'output_format_output_format_output_format_03',
    'format_convert_format_convert_mix',
    'format_convert_format_convert_multi',
    'format_convert_format_convert_single',
    'format_convert_format_convert_transfer'
]

order_fields = [
    'character_order_keep_order_character',
    'character_order_reversed_order_character',
    'character_order_specify_order_character',
    'sentence_order_keep_order_sentence',
    'sentence_order_reversed_order_sentence',
    'word_order_keep_order_word',
    'word_order_reversed_order_word',
    'word_order_specify_order_word',
    'check_order_check_order_character',
    'check_order_check_order_word'
]

statistics_fields = [
    'relation_analysis_generate_statistic_relation'
]

list_mapping_fields = [
    'navigation_and_count_count_or_navigation_count-easy',
    'navigation_and_count_count_or_navigation_count-middle',
    'navigation_and_count_count_or_navigation_navigation-easy',
    'navigation_and_count_count_or_navigation_navigation-middle'
]

# ------------------------------------------------------
# 主逻辑
# ------------------------------------------------------

def process_model_scores(csv_path, output_path=None):
    # 读取CSV
    df = pd.read_csv(csv_path)
    results = []

    # 遍历每一行（每个模型）
    for idx, row in df.iterrows():
        model_name = row['model_name']

        # 计算每一组的得分（取所有子任务的平均）
        def safe_average(fields):
            values = [row[field] for field in fields if field in row and pd.notna(row[field])]
            return sum(values) / len(values) if values else None

        unstructured = safe_average(unstructured_fields)
        structured = safe_average(structured_fields)
        format_score = safe_average(format_fields)
        order = safe_average(order_fields)
        statistics = safe_average(statistics_fields)
        list_mapping = safe_average(list_mapping_fields)

        # 计算最终Average（六项的平均）
        metric_components = [x for x in [unstructured, structured, format_score, order, statistics, list_mapping] if x is not None]
        if metric_components:
            overall_avg = sum(metric_components) / len(metric_components)
        else:
            overall_avg = None

        results.append([
            model_name,
            unstructured,
            structured,
            format_score,
            order,
            statistics,
            list_mapping,
            overall_avg
        ])

    # 整理成DataFrame
    columns = [
        'Model',
        'Unstructured Text',
        'Structured Text',
        'Format',
        'Order',
        'Statistics',
        'List Mapping',
        'Average'
    ]
    result_df = pd.DataFrame(results, columns=columns)
    print(result_df)

    # 保存为Excel
    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"Saved results to {output_path}")

    return result_df

# ------------------------------------------------------
# argparse 命令行接口
# ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model evaluation results.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    args = parser.parse_args()

    input_csv = args.input

    # 自动生成输出文件名：加 _final.xlsx
    base_name = os.path.splitext(input_csv)[0]
    output_excel = f"{base_name}_final.xlsx"

    process_model_scores(input_csv, output_excel)
