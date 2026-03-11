"""
PRISM Inference Script (Open-Source Models)
Batch inference for evaluating open-source LLMs on the PRISM benchmark
using vLLM for efficient serving.
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str          # Short name for output files
    path: str          # Local path or HuggingFace model ID
    display_name: str  # Display name for logging


# ============================================================
# Configure your models here
# ============================================================
MODELS = [
    ModelConfig(
        name="qwen2.5-72b",
        path="Qwen/Qwen2.5-72B-Instruct",
        display_name="Qwen2.5-72B-Instruct"
    ),
    ModelConfig(
        name="llama3.3-70b",
        path="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama-3.3-70B-Instruct"
    ),
]

# Inference hyperparameters
BATCH_SIZE = 64
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 2048


def get_num_gpus() -> int:
    """Detect the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def load_all_datasets(base_dir: str) -> List[Dict]:
    """
    Recursively load all JSON dataset files from the PRISM data directory.
    Each item is annotated with category and subcategory metadata.
    """
    all_data = []
    json_files = glob.glob(f"{base_dir}/**/*.json", recursive=True)

    print(f"Found {len(json_files)} JSON files")

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            category = Path(json_file).parent.name
            subcategory = Path(json_file).stem

            for item in data:
                item['category'] = category
                item['subcategory'] = subcategory
                item['source_file'] = json_file
                all_data.append(item)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Total loaded: {len(all_data)} samples")
    return all_data


def build_prompt(question: str, model_type: str = "qwen") -> str:
    """
    Build chat prompt for the given model type.

    Args:
        question: The evaluation question text.
        model_type: One of 'qwen' or 'llama'.

    Returns:
        Formatted prompt string.
    """
    system_msg = (
        "You are a helpful assistant. Please follow the instructions "
        "in the user's query carefully and provide accurate responses."
    )

    if model_type == "qwen":
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_type == "llama":
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def run_inference_for_model(
    model_config: ModelConfig,
    dataset: List[Dict],
    tensor_parallel_size: int,
    model_type: str = "qwen"
) -> List[Dict]:
    """
    Run batch inference for a single model on the entire dataset.

    Args:
        model_config: Model configuration.
        dataset: List of evaluation samples.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        model_type: Prompt template type ('qwen' or 'llama').

    Returns:
        List of result dicts with model outputs.
    """
    print(f"\n{'='*80}")
    print(f"Running model: {model_config.display_name}")
    print(f"Model path: {model_config.path}")
    print(f"{'='*80}\n")

    # Initialize vLLM engine
    llm = LLM(
        model=model_config.path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>", "<|eot_id|>"]
    )

    # Build prompts
    prompts = [build_prompt(item['question'], model_type) for item in dataset]

    results = []
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Starting inference: {len(prompts)} samples, {total_batches} batches...")

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Inference"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_data = dataset[i:i + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            result = {
                'id': batch_data[j]['id'],
                'ID': batch_data[j].get('ID', batch_data[j]['id']),
                'category': batch_data[j]['category'],
                'subcategory': batch_data[j]['subcategory'],
                'question': batch_data[j]['question'],
                'expected_answer': batch_data[j].get('answer', ''),
                'model_output': generated_text,
                'model_name': model_config.name,
                'metric': batch_data[j].get('metric', '')
            }
            results.append(result)

    print(f"\nModel {model_config.display_name} inference complete!")

    # Free GPU memory
    del llm
    torch.cuda.empty_cache()

    return results


def save_results(results: List[Dict], model_name: str, output_dir: str = "./results"):
    """Save inference results to JSONL format with a summary file."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{model_name}_results.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Results saved to: {output_file} ({len(results)} samples)")

    # Generate summary statistics
    summary_file = os.path.join(output_dir, f"{model_name}_summary.txt")
    category_stats = {}
    for result in results:
        cat = result['category']
        category_stats[cat] = category_stats.get(cat, 0) + 1

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total samples: {len(results)}\n\n")
        f.write("Category statistics:\n")
        for cat, count in sorted(category_stats.items()):
            f.write(f"  {cat}: {count}\n")

    print(f"Summary saved to: {summary_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRISM Benchmark Inference (Open-Source Models)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to PRISM data directory")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--model-type", type=str, default="qwen",
                        choices=["qwen", "llama"],
                        help="Prompt template type")
    args = parser.parse_args()

    num_gpus = args.gpus or get_num_gpus()

    print("=" * 80)
    print("PRISM Benchmark Inference")
    print("=" * 80)
    print(f"GPUs: {num_gpus} | Batch size: {BATCH_SIZE}")
    print(f"Temperature: {TEMPERATURE} | Top-p: {TOP_P}")
    print(f"Models: {len(MODELS)}")
    print("=" * 80)

    # Load dataset
    dataset = load_all_datasets(args.data_dir)
    if not dataset:
        print("Error: No data found!")
        return

    # Run inference for each model
    for model_config in MODELS:
        results = run_inference_for_model(
            model_config, dataset, num_gpus, args.model_type
        )
        save_results(results, model_config.name, args.output_dir)

    print("\n" + "=" * 80)
    print("All models completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
