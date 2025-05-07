import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModel,
)
# Assuming modeling_phi is correctly importable in your environment
# If not, adjust the import path as needed
try:
    from modeling_phi.modeling_mixformer_sequential import MixFormerSequentialForCausalLM
except ImportError:
    print("Warning: Could not import MixFormerSequentialForCausalLM. 'phi' models might not load correctly.")
    MixFormerSequentialForCausalLM = None # Define it as None to avoid NameErrors later

import tasks # Assuming tasks is correctly importable
from tqdm import tqdm
import json
import os
import argparse # Import argparse
import datetime # Import datetime

def _load_tokenizer(tok_path, name):
    """Loads tokenizer and sets default special tokens if missing."""
    print(f"Loading tokenizer from: {tok_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        print("Testing tokenizer...")
        # Use a simple string for encoding test
        test_string = "hello\n\nhello"
        print(f"'{test_string}': ", tokenizer.encode(test_string))
        print("bos_token_id: {}, eos_token_id: {}, pad_token_id: {}".format(
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id))

        # Set default token IDs only if they are None or missing
        # Common defaults: pad=0, bos=1, eos=2 (check model specifics if needed)
        if "chatglm" in name.lower() or "qwen" in name.lower():
            # These models often have specific token handling
            pass
        else:
            if tokenizer.pad_token_id is None:
                # Common practice is to use eos_token_id if pad_token is missing
                if tokenizer.eos_token_id is not None:
                   tokenizer.pad_token_id = tokenizer.eos_token_id
                   print(f"Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id}), now pad_token is {tokenizer.decode(tokenizer.pad_token_id)}")
                else:
                   # Fallback if EOS is also missing (less common)
                   tokenizer.pad_token_id = 0
                   print(f"Setting pad_token_id to {tokenizer.pad_token_id}, now pad_token is {tokenizer.decode(tokenizer.pad_token_id)}")

            if tokenizer.bos_token_id is None:
                # Setting BOS might be less critical depending on the model/task
                # Use a common default if truly needed, but be cautious
                # tokenizer.bos_token_id = 1
                # print(f"Setting bos_token_id to {tokenizer.bos_token_id}, now bos_token is {tokenizer.decode(tokenizer.bos_token_id)}")
                pass # Often models handle BOS internally or don't require explicit setting here

            if tokenizer.eos_token_id is None:
                 # Setting EOS might be less critical depending on the model/task
                 # Use a common default if truly needed, but be cautious
                 # tokenizer.eos_token_id = 2
                 # print(f"Setting eos_token_id to {tokenizer.eos_token_id}, now eos_token is {tokenizer.decode(tokenizer.eos_token_id)}")
                 pass # Often models handle EOS internally

        # Default padding side for generation
        tokenizer.padding_side = 'left'
        print("Setting padding_side to left")
        print("Vocab length: %d" % len(tokenizer))
        return tokenizer

    except Exception as e:
        print(f"Error loading tokenizer from {tok_path}: {e}")
        raise # Re-raise the exception to stop execution if tokenizer fails

# Updated function signature to accept model_path and model_name
def load_model_and_tokenizer(model_path, model_name):
    """Loads the specified model and its tokenizer."""
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA device: cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")


    print("="*50)
    print(f"Loading model '{model_name}' from path: <{model_path}>")
    print("="*50)

    # load tokenizer
    tokenizer = _load_tokenizer(model_path, model_name)

    # load model
    print(f"\nLoading model '{model_name}'...")
    model_load_args = {"pretrained_model_name_or_path": model_path, "trust_remote_code": True}
    model = None
    dtype = torch.float16 # Default dtype

    try:
        if "72b" in model_name.lower() or "65b" in model_name.lower() or "70b" in model_name.lower() or "8x7b" in model_name.lower(): # Added 8x7B here
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            print("Using accelerate for large model loading (device_map='auto')...")
            # Large models often require device_map="auto" and potentially quantization
            # init_empty_weights might be needed if memory is very tight before dispatch
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_config(
            #         AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            #     ) # Load config first if using init_empty_weights

            # Recommended approach for multi-GPU or large models:
            model = AutoModelForCausalLM.from_pretrained(
                **model_load_args,
                device_map="auto",
                torch_dtype=dtype # Specify dtype for accelerate
                # Add other accelerate args like load_in_8bit=True, load_in_4bit=True if needed
            )
            # Tying weights is usually done internally by from_pretrained if applicable
            # model.tie_weights() # Generally not needed unless manually constructing parts
            model.eval() # Set to evaluation mode

        elif "phi" in model_name.lower():
            print(f"Loading Phi model using AutoModelForCausalLM, using dtype {dtype} and device_map='auto'...")
            model = AutoModelForCausalLM.from_pretrained(
                **model_load_args,
                device_map="auto",
                torch_dtype=dtype
            )
            model.eval()

        elif "chatglm" in model_name.lower():
             # ChatGLM often uses AutoModel, not AutoModelForCausalLM
            print(f"Loading AutoModel (ChatGLM) model, using dtype {dtype} and device {device}...")
            model = AutoModel.from_pretrained(**model_load_args)
            # ChatGLM might require specific dtype handling or device placement
            model.eval().to(dtype=dtype, device=device)

        elif "baichuan" in model_name.lower() or "qwen" in model_name.lower():
            # These models might prefer bfloat16 if available and supported
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Loading AutoModelForCausalLM ({model_name}), using dtype {compute_dtype} and device {device}...")
            model = AutoModelForCausalLM.from_pretrained(**model_load_args)
            model.eval().to(dtype=compute_dtype, device=device)

        else: # Default case for other CausalLM models
            print(f"Loading default AutoModelForCausalLM, using dtype {dtype} and device {device}...")
            model = AutoModelForCausalLM.from_pretrained(**model_load_args)
            model.eval().to(dtype=dtype, device=device)

        print("Model loading success...")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model {model_name} from {model_path}: {e}")
        # Clean up potentially loaded model parts if using accelerate/device_map
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise # Re-raise the exception

@torch.no_grad()
def greedy_until(requests, model, tokenizer, config, disable_tqdm=False, model_max_length=2048):
    """
    Generates text for requests using the model.

    Args:
        requests: List of dictionaries, each with a 'context' key.
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.
        config: Dictionary with generation parameters ('batch_size', 'max_gen_len', 'end_token', 'is_sample').
        disable_tqdm: Whether to disable the progress bar.
        model_max_length: The maximum sequence length the model can handle.

    Returns:
        List of generated response strings.
    """
    # step 0: unpack config
    batch_size = config.get('batch_size', 4) # Default batch size if not provided
    max_gen_len = config.get('max_gen_len', 50)
    end_token = config.get('end_token', None) # Specific string to stop generation
    is_sample = config.get('is_sample', False) # Whether to use sampling

    # Prepare requests for processing
    new_reqs = []
    for idx, request in enumerate(requests):
        context = request['context']
        # Be careful with encoding/decoding if context contains special tokens
        # It's generally safer to work with token IDs directly for truncation
        context_enc = tokenizer.encode(context, add_special_tokens=False)

        # Truncate from the left to fit within model_max_length minus generation space
        max_context_len = model_max_length - max_gen_len - 5 # Add a small buffer
        if len(context_enc) > max_context_len:
             context_enc = context_enc[-max_context_len:]
             print(f"Warning: Request {idx} context truncated to {max_context_len} tokens.")

        # Decode back the potentially truncated context for use as key/prompt later
        # Ensure clean decoding, handling potential tokenization artifacts if necessary
        clean_context = tokenizer.decode(context_enc, skip_special_tokens=True) # Use skip_special_tokens carefully

        # Store original context (key), processed context (clean_key), and length for sorting
        key = context # Original context as identifier
        length = -len(context_enc) # Sort by length descending (longest first)
        new_reqs.append({'key': key, 'clean_key': clean_context, 'length': length, 'orig_index': idx})

    # step 1: sort by length (longest first) to potentially optimize padding
    new_reqs.sort(key=lambda x: x['length'])
    print(f"Processing {len(new_reqs)} requests. Max context tokens (approx): {-new_reqs[0]['length']}, Min context tokens (approx): {-new_reqs[-1]['length']}")

    # step 2: batch processing
    cache_res = {} # Use original index to store results
    for i in tqdm(range(0, len(new_reqs), batch_size), disable=disable_tqdm, desc="Generating"):
        chunk = new_reqs[i:i + batch_size]
        clean_keys = [x['clean_key'] for x in chunk] # Prompts for the model

        # step 3: padding and tokenization
        # Tokenize the batch with padding and truncation
        inputs = tokenizer(
            clean_keys,
            return_tensors='pt',
            padding="longest", # Pad to the longest sequence in the batch
            max_length=model_max_length - max_gen_len, # Ensure space for generation
            truncation=True, # Truncate if needed (should already be handled, but good safety)
            return_attention_mask=True,
            add_special_tokens=True # Add BOS/EOS if the model expects them
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        current_input_length = input_ids.size(1)
        # Calculate max_new_tokens carefully
        max_new_tokens = min(max_gen_len, model_max_length - current_input_length)

        if max_new_tokens <= 0:
             print(f"Warning: Batch starting at index {i} has no room for generation (input_len={current_input_length}, max_len={model_max_length}). Skipping generation for this batch.")
             # Assign empty strings or a placeholder error message
             for req_data in chunk:
                 cache_res[req_data['orig_index']] = "[ERROR: No space for generation]"
             continue # Skip to the next batch

        # step 4: call model
        eos_token_id_list = []
        if end_token:
             # Handle multiple EOS tokens if needed, e.g., ["\n", "END"]
             end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
             if end_token_ids:
                 eos_token_id_list.extend(end_token_ids)
        # Also include the default EOS token if it exists
        if tokenizer.eos_token_id is not None:
            # Avoid duplicates if end_token is the same as tokenizer's eos_token
            if tokenizer.eos_token_id not in eos_token_id_list:
                 eos_token_id_list.append(tokenizer.eos_token_id)

        # If no specific EOS tokens, use only the tokenizer's default if available
        eos_token_param = eos_token_id_list if eos_token_id_list else tokenizer.eos_token_id


        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_param, # Can be a list or single int
            "do_sample": is_sample,
            # Add sampling parameters if is_sample is True
            # "temperature": 0.7,
            # "top_p": 0.9,
            # "top_k": 50,
             "use_cache": True, # Generally beneficial for speed
             "attention_mask": attention_mask.to(model.device) # Pass attention mask
        }

        try:
            outputs = model.generate(
                input_ids.to(model.device),
                **generate_kwargs
            )
        except Exception as e:
            print(f"\nError during model.generate for batch starting at index {i}: {e}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Kwargs: {generate_kwargs}")
            # Assign error messages to results for this batch
            for req_data in chunk:
                cache_res[req_data['orig_index']] = f"[ERROR: Generation failed: {e}]"
            continue # Skip to the next batch

        # step 5: decode and extract generated part
        # Decode the full output sequences
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the newly generated text
        for req_data, input_text, full_output_text in zip(chunk, clean_keys, output_texts):
            # Find the generated part by removing the input prompt
            # Be robust to potential minor variations in decoding/tokenization
            # A common way is to remove the input text from the start
            if full_output_text.startswith(input_text):
                response = full_output_text[len(input_text):].strip()
            else:
                # Fallback: try splitting (less reliable if input appears in output)
                # response = full_output_text.split(input_text, 1)[-1].strip()
                # Or just return the full output with a warning if prompt removal fails
                print(f"Warning: Could not cleanly remove prompt for request index {req_data['orig_index']}. Prompt: '{input_text[:50]}...', Output: '{full_output_text[:100]}...'")
                response = full_output_text # Return full output as a fallback
            cache_res[req_data['orig_index']] = response

    # Reconstruct results in the original order
    results = [cache_res[i] for i in range(len(requests))]
    print(f"Generated {len(results)} results.")

    return results

# Updated function signature to accept model_path and model_name
def run_one_model(tasks_list, model_path, model_name, output_base_dir):
    """Runs evaluation tasks for a single model."""
    print(f"\n===== Running evaluation for model: {model_name} =====")
    # get dataset definitions
    task_dict = {}
    try:
        for task_name in tasks_list:
            if task_name in tasks.TASK_REGISTRY:
                 task_dict[task_name] = tasks.TASK_REGISTRY[task_name]()
            else:
                 print(f"Warning: Task '{task_name}' not found in TASK_REGISTRY. Skipping.")
    except AttributeError:
         print("Error: tasks.TASK_REGISTRY not found or not structured as expected.")
         return {} # Return empty metrics if tasks cannot be loaded

    if not task_dict:
        print("No valid tasks found to run.")
        return {}

    # config for tasks
    task_config = {name: task.get_config() for name, task in task_dict.items()}

    # get requests for each task
    task_requests = {}
    for task_name, task in task_dict.items():
        try:
            task_requests[task_name] = task.construct_requests()
            print(f"Built {len(task_requests[task_name])} requests for task <{task_name}>.")
        except Exception as e:
             print(f"Error constructing requests for task {task_name}: {e}")
             # Optionally remove the task if requests fail
             # del task_dict[task_name]
             # del task_config[task_name]


    # load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, model_name)
    except Exception as e:
        print(f"Failed to load model or tokenizer for {model_name}. Skipping evaluation. Error: {e}")
        return {} # Return empty metrics if loading fails


    # Determine model_max_length based on model config or name heuristics
    model_max_length = 2048 # Default
    try:
        if hasattr(model.config, 'max_position_embeddings'):
            model_max_length = model.config.max_position_embeddings
            print(f"Model max length from config: {model_max_length}")
        # Fallback heuristics if config attribute is missing
        elif "llama2" in model_name.lower() or "baichuan2" in model_name.lower() or "llama-3" in model_name.lower() or "llama3" in model_name.lower():
            model_max_length = 4096
            print(f"Using heuristic max length for {model_name}: {model_max_length}")
        elif "qwen" in model_name.lower():
             # Qwen models vary, check specific model card if possible
             # model_max_length = 8192 # Older Qwen
             model_max_length = 32768 # Newer Qwen/Qwen2 often have larger context
             print(f"Using heuristic max length for {model_name}: {model_max_length}")
        else:
            print(f"Using default max length: {model_max_length}")
    except Exception as e:
        print(f"Warning: Could not determine model_max_length reliably ({e}). Using default: {model_max_length}")


    # Run evaluation for each task
    return_metrics = {}
    for task_name, task in task_dict.items():
        if task_name not in task_requests:
            print(f"Skipping task '{task_name}' due to earlier request construction error.")
            continue # Skip if requests failed

        print(f"\n--- Testing task: {task_name} ---")
        current_task_requests = task_requests[task_name]
        current_task_config = task_config[task_name]

        # Run generation
        results = greedy_until(
            current_task_requests,
            model,
            tokenizer,
            current_task_config,
            disable_tqdm=False,
            model_max_length=model_max_length
        )

        # Process results and calculate metrics
        try:
            metrics, logs = task.process_results(results)
            print(f"Metrics for {task_name}: ", end="")
            for key, value in metrics.items():
                metric_key = f"{task_name}_{key}" # Prefix metric name with task name
                print(f"{key}: {value:.4f}", end=", ")
                return_metrics[metric_key] = value
            print() # Newline after metrics

            # Write logs to file
            log_dir = os.path.join(output_base_dir, task_name)
            # Sanitize model name for filename
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            log_path = os.path.join(log_dir, f"{safe_model_name}.jsonl") # Use .jsonl for line-delimited json

            try:
                os.makedirs(log_dir, exist_ok=True)
                with open(log_path, 'w', encoding="utf8") as f:
                    for log_entry in logs:
                         # Ensure log entries are serializable
                         try:
                             f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                         except TypeError as json_err:
                             print(f"Warning: Could not serialize log entry for {task_name}: {json_err}. Entry: {log_entry}")
                             f.write(json.dumps({"error": "Serialization failed", "original_data": str(log_entry)}, ensure_ascii=False) + '\n')
                print(f"Saved logs for {task_name} to {log_path}")
            except IOError as e:
                print(f"Error writing logs for task {task_name} to {log_path}: {e}")

        except Exception as e:
            print(f"Error processing results for task {task_name}: {e}")
            # Add placeholder metrics to indicate failure
            return_metrics[f"{task_name}_processing_error"] = 1.0


    # Clean up GPU memory after processing a model
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")

    print(f"===== Finished evaluation for model: {model_name} =====")
    return return_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluations on specified tasks.")
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="Comma-separated list of paths to the model directories (Hugging Face format)."
    )
    parser.add_argument(
        "--model_names",
        type=str,
        required=True,
        help="Comma-separated list of names corresponding to the model paths (used for internal logic and logging)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../logs", # Default relative log directory
        help="Base directory to save task logs and the final metrics CSV."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="dict_search_string,dict_search_number,natural_language_string,check_order,character_order,word_order,sentence_order,check_dedup,character_dedup,word_dedup,sentence_dedup,relation_analysis,navigation_and_count,check_format,output_format,format_convert,list_number",
        help="Comma-separated list of task names to run."
    )

    args = parser.parse_args()

    # Split the comma-separated strings into lists
    model_paths_list = [path.strip() for path in args.model_paths.split(',')]
    model_names_list = [name.strip() for name in args.model_names.split(',')]
    tasks_list = [task.strip() for task in args.tasks.split(',')]

    # Validate that the number of paths matches the number of names
    if len(model_paths_list) != len(model_names_list):
        raise ValueError("The number of model paths must match the number of model names.")

    if not tasks_list or all(s == '' for s in tasks_list):
         raise ValueError("No tasks specified. Use the --tasks argument.")


    # --- Define tasks --- (or keep the default list from args)
    # You can override tasks_list here if needed, but using args.tasks is more flexible
    # tasks_list = [
    #     "dict_search_string",
    #     # ... other tasks ...
    # ]
    print(f"Selected tasks: {tasks_list}")

    # --- Run evaluations ---
    all_metrics = []
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True) # Ensure base output dir exists

    # Use zip to iterate through paths and names together
    for model_path, model_name in zip(model_paths_list, model_names_list):
        print(f"\n>>> Processing Model: {model_name} from {model_path}")
        # Pass the output base directory to the run function
        metrics = run_one_model(tasks_list, model_path, model_name, output_base_dir)
        if metrics: # Only append if metrics were successfully generated
            # Add model name to the metrics dict for easier identification later
            metrics["model_name"] = model_name
            all_metrics.append(metrics)
        else:
            print(f"Skipping metrics for {model_name} due to errors during evaluation.")

    # --- Write combined results to CSV ---
    if not all_metrics:
        print("\nNo metrics collected. CSV file will not be generated.")
    else:
        time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") # Use H M S for uniqueness
        csv_filename = f"evaluation_results_{time_tag}.csv"
        csv_path = os.path.join(output_base_dir, csv_filename)

        print(f"\nWriting combined metrics to: {csv_path}")

        try:
            # Dynamically get all unique metric keys from all runs
            # Ensure consistent ordering, e.g., alphabetical
            header_keys = set()
            for metrics in all_metrics:
                header_keys.update(metrics.keys())
            # Make 'model_name' the first column, then sort the rest
            sorted_headers = ['model_name'] + sorted([k for k in header_keys if k != 'model_name'])


            with open(csv_path, "w", encoding='utf8', newline='') as f:
                # Write header
                f.write(",".join(sorted_headers) + "\n")

                # Write data rows
                for metrics in all_metrics:
                    row_values = [f"{metrics.get(key, 'N/A'):.4f}" if isinstance(metrics.get(key), float) else str(metrics.get(key, 'N/A')) for key in sorted_headers]
                    f.write(",".join(row_values) + "\n")

            print("Successfully wrote metrics CSV.")

        except IOError as e:
            print(f"Error writing metrics CSV to {csv_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while writing the CSV: {e}")

    print("\nEvaluation finished.")