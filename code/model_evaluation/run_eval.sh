source activate own_ICL

# Define model paths and their corresponding names in arrays
declare -a model_paths=(
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-3-small-128k-instruct/snapshots/ad85cab62be398dc90203c4377a4ccbf090fbb36"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-3-medium-128k-instruct/snapshots/fa7d2aa4f5ea69b2e36b20d050cdae79c9bfbb3f"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/phi_test/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    # "/home/v-huzhengyu/zhengyu_blob_home/hugging_face_models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
)

declare -a model_names=(
    # "Qwen2.5-72B-Chat"
    # "Qwen2.5-32B-Chat"
    # "Qwen2.5-14B-Chat"
    # "Mistral-7B-v0.3"
    # "Phi-4-mini"
    # "Phi-3-small"
    # "Phi-3-medium"
    # "Phi-3-mini"
    "Llama-3.1-70B"
    # "Llama-3.1-8B"
    # "DeepSeek-V2-Lite-Chat"
    # "LLaMA-3-8B"
)

# Run evaluation for each model
for i in "${!model_paths[@]}"; do
    echo "Evaluating ${model_names[$i]}..."
    python run_icl_eval.py \
        --model_paths="${model_paths[$i]}" \
        --model_names="${model_names[$i]}"
    echo "Finished evaluating ${model_names[$i]}"
    echo "----------------------------------------"
done

bash /home/aiscuser/zhengyu_blob_home/kkk_vllm_first_4.sh