"""
PRISM Inference Script (Proprietary API Models)
Inference for evaluating proprietary LLMs via OpenAI-compatible APIs
on the PRISM benchmark.
"""

import json
import os
import glob
import time
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Inference Hyperparameters
# ============================================================
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_RETRIES = 3
TIMEOUT = 60


@dataclass
class APIModelConfig:
    """Configuration for a single API model."""
    provider: str
    model_name: str
    api_key: str
    base_url: str


def load_model_configs(config_file: str) -> List[APIModelConfig]:
    """
    Load model configurations from a JSON config file.

    Expected format:
    {
        "base_url": "https://api.example.com/v1",
        "models": [
            {
                "provider": "openai",
                "api_key": "sk-xxx",
                "model_names": ["gpt-4o", "gpt-5.1"]
            }
        ]
    }
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    base_url = config['base_url']
    models = []

    for provider_config in config['models']:
        provider = provider_config['provider']
        api_key = provider_config['api_key']
        for model_name in provider_config['model_names']:
            models.append(APIModelConfig(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url
            ))

    return models


def load_all_datasets(base_dir: str) -> List[Dict]:
    """Recursively load all JSON dataset files from the PRISM data directory."""
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


def call_api_with_retry(client: OpenAI, model_name: str, messages: List[Dict]) -> str:
    """
    Call the API with automatic retry on failure.

    Args:
        client: OpenAI client instance.
        model_name: Model identifier.
        messages: Chat messages in OpenAI format.

    Returns:
        Model response text, or error string on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Retry in {wait_time}s... Error: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                print(f"  Max retries reached: {str(e)[:100]}")
                return f"[ERROR] {str(e)}"


def run_inference_for_model(
    model_config: APIModelConfig,
    dataset: List[Dict]
) -> List[Dict]:
    """Run inference for a single API model on the entire dataset."""
    print(f"\n{'='*80}")
    print(f"Running: {model_config.provider} / {model_config.model_name}")
    print(f"{'='*80}\n")

    client = OpenAI(api_key=model_config.api_key, base_url=model_config.base_url)
    results = []

    system_prompt = (
        "You are a helpful assistant. Please follow the instructions "
        "in the user's query carefully and provide accurate responses."
    )

    for item in tqdm(dataset, desc="Inference"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['question']}
        ]

        output = call_api_with_retry(client, model_config.model_name, messages)

        result = {
            'id': item['id'],
            'ID': item.get('ID', item['id']),
            'category': item['category'],
            'subcategory': item['subcategory'],
            'question': item['question'],
            'expected_answer': item.get('answer', ''),
            'model_output': output,
            'model_name': model_config.model_name,
            'provider': model_config.provider,
            'metric': item.get('metric', '')
        }
        results.append(result)

    print(f"\n{model_config.model_name} inference complete!")
    return results


def save_results(results: List[Dict], model_config: APIModelConfig, output_dir: str):
    """Save results in JSONL format with summary statistics."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_config.model_name.replace('/', '_').replace(':', '_')
    output_file = os.path.join(output_dir, f"{model_config.provider}_{safe_name}_results.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Results saved to: {output_file} ({len(results)} samples)")


def main():
    parser = argparse.ArgumentParser(description="PRISM Benchmark Inference (API Models)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON file")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to PRISM data directory")
    parser.add_argument("--output-dir", type=str, default="./results/api",
                        help="Directory to save results")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Specific model names to run (default: all)")
    args = parser.parse_args()

    print("=" * 80)
    print("PRISM Benchmark Inference (API Models)")
    print("=" * 80)

    # Load models
    all_models = load_model_configs(args.config)
    models = (
        [m for m in all_models if m.model_name in args.models]
        if args.models else all_models
    )

    print(f"Models to run: {len(models)}")
    for m in models:
        print(f"  - {m.provider}: {m.model_name}")

    # Load dataset
    dataset = load_all_datasets(args.data_dir)
    if not dataset:
        print("Error: No data found!")
        return

    # Run inference
    for model_config in models:
        results = run_inference_for_model(model_config, dataset)
        save_results(results, model_config, args.output_dir)

    print("\n" + "=" * 80)
    print("All models completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
