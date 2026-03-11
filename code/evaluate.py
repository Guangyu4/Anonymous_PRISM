"""
PRISM Evaluation Script
Automated evaluation of model outputs against ground-truth answers
using an LLM-as-judge approach.
"""

import json
import os
import glob
import argparse
from typing import List, Dict
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams


# ============================================================
# Evaluation Configuration
# ============================================================
BATCH_SIZE = 32
TENSOR_PARALLEL_SIZE = 4

JUDGE_PROMPT_TEMPLATE = """You are a strict answer evaluator. Compare two answers to determine if they express EXACTLY the same concept.

Ground Truth Answer: {standard_answer}
Model Prediction: {generated_answer}

Evaluation Logic (Reasonable Mode):

1. Ignore "Container" Words:
   - "Punk Rock" (True) == "Punk Rock Band" (Pred) -> TRUE
   - "Drama Movie" (True) == "Drama Film" (Pred) -> TRUE

2. Handle Synonyms & Abbreviations:
   - "Indie" = "Independent"
   - "Film" = "Movie" = "Motion Picture"

3. Allow Specificity & Sub-genres:
   - Ground Truth: "Comedy" vs Model: "Romantic Comedy" -> TRUE
   - Ground Truth: "Novel" vs Model: "Historical Novel" -> TRUE

4. Partial Matches in Lists:
   - If Ground Truth lists multiple items and Model picks the most prominent one -> TRUE

5. When to return FALSE:
   - Only if the answers are contradictory or unrelated.

Output: Answer ONLY with "true" or "false".

Your answer:"""


def truncate_answer(answer: str, max_chars: int = 500) -> str:
    """Truncate overly long answers for evaluation."""
    answer = answer.strip()
    if len(answer) > max_chars:
        return answer[:max_chars] + "..."
    return answer


def build_judge_prompt(generated_answer: str, standard_answer: str, model_type: str = "llama") -> str:
    """
    Build the evaluation prompt for the judge model.

    Args:
        generated_answer: The model's output to evaluate.
        standard_answer: The ground-truth answer.
        model_type: Prompt template type ('llama' or 'qwen').

    Returns:
        Formatted prompt string for the judge model.
    """
    question = JUDGE_PROMPT_TEMPLATE.format(
        standard_answer=truncate_answer(standard_answer),
        generated_answer=truncate_answer(generated_answer)
    )

    system_msg = "You are a helpful assistant. Please follow the instructions carefully and provide accurate responses."

    if model_type == "llama":
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "qwen":
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_batch(llm: LLM, data_batch: List[Dict], sampling_params: SamplingParams,
                   model_type: str = "llama") -> List[bool]:
    """
    Evaluate a batch of samples using the judge model.

    Returns:
        List of boolean labels (True = correct, False = incorrect/hallucinated).
    """
    prompts = []
    for item in data_batch:
        generated = item.get("answer", "").strip()
        standard = item.get("Answer", "").strip()
        prompts.append(build_judge_prompt(generated, standard, model_type))

    outputs = llm.generate(prompts, sampling_params)

    labels = []
    for output in outputs:
        response = output.outputs[0].text.strip().lower()
        if "true" in response and "false" not in response:
            labels.append(True)
        elif "false" in response:
            labels.append(False)
        else:
            labels.append(False)  # Default to incorrect if uncertain

    return labels


def process_file(input_file: str, llm: LLM, sampling_params: SamplingParams,
                 output_dir: str, model_type: str = "llama") -> Dict:
    """
    Evaluate all samples in a single data file.

    Returns:
        Summary dict with accuracy statistics.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    filename = os.path.basename(input_file)
    print(f"\nEvaluating: {filename} ({total} samples)")

    correct = 0
    for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Processing {filename}"):
        batch = data[i:i + BATCH_SIZE]
        labels = evaluate_batch(llm, batch, sampling_params, model_type)

        for j, label in enumerate(labels):
            batch[j]["label"] = label
            if label:
                correct += 1

    accuracy = correct / total if total > 0 else 0

    # Save evaluated results
    os.makedirs(output_dir, exist_ok=True)
    output_filename = filename.replace('.json', '_evaluated.json')
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Saved to: {output_file}")

    return {
        'filename': filename,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Evaluation Script")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing result JSON files to evaluate")
    parser.add_argument("--output-dir", type=str, default="./evaluated",
                        help="Directory to save evaluated results")
    parser.add_argument("--judge-model", type=str, required=True,
                        help="Path to the judge model")
    parser.add_argument("--model-type", type=str, default="llama",
                        choices=["llama", "qwen"],
                        help="Judge model prompt template type")
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()

    # Find all result files
    result_files = sorted(glob.glob(os.path.join(args.data_dir, "result_*.json")))
    result_files = [f for f in result_files if '_evaluated' not in f]

    if not result_files:
        print(f"Error: No result_*.json files found in {args.data_dir}")
        return

    print(f"Found {len(result_files)} files to evaluate")

    # Initialize judge model
    print(f"Loading judge model: {args.judge_model}")
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.gpus,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=10,
        stop=["<|eot_id|>", "<|im_end|>"]
    )

    # Evaluate each file
    results = []
    for input_file in result_files:
        try:
            result = process_file(input_file, llm, sampling_params, args.output_dir, args.model_type)
            results.append(result)
        except Exception as e:
            print(f"Error processing {os.path.basename(input_file)}: {e}")

    # Free GPU memory
    del llm
    torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"{'File':<35} {'Accuracy':<12} {'Correct/Total'}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['filename']:<35} {r['accuracy']:>6.2%}      {r['correct']:>3}/{r['total']:<3}")

    avg_acc = sum(r['accuracy'] for r in results) / len(results) if results else 0
    print(f"{'-'*80}")
    print(f"{'Average:':<35} {avg_acc:>6.2%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
