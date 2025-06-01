import argparse
import json
import os
import torch
import pickle
from vllm import LLM, SamplingParams
from utils.data_loaders import (
    load_geo3k_dataset,
    load_logicvista_dataset,
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_dynamath_dataset,
    load_olympiadbench_dataset,
    load_mathvision_dataset,
    load_hallubench_dataset
)
from utils.processing import (
    prepare_prompts,
    process_outputs,
    calculate_metrics
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    
    # Model and runtime parameters
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--eval-threads", type=int, default=32, help="Number of threads for evaluation")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    
    # Dataset selection
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'")
    
    # Dataset-specific paths
    parser.add_argument("--data-path", type=str, default="./data", help="")
    parser.add_argument("--tsv-data-path", type=str, default=None, help="")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="eval", choices=["inference", "eval", "all"], 
                        help="Operation mode: 'inference' for generation only, 'eval' for evaluation only, 'all' for both")
    
    return parser.parse_args()

def get_pickle_path(output_dir, dataset_name):
    """Generate the path for saving/loading pickle files"""
    return os.path.join(output_dir, f"{dataset_name}_outputs.pkl")

def inference_mode(all_samples, args):
    """Run only the inference step and save outputs"""
    # Initialize model
    print(f"Initializing model from {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.7,
        limit_mm_per_prompt={"image": 10},
        max_model_len=args.max_model_len
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # Process each dataset
    for dataset_name, samples in all_samples.items():
        print(f"Running inference on {dataset_name}...")
        prompts, metadata = prepare_prompts(dataset_name, samples, args)
        
        # Generate outputs
        outputs = llm.generate(prompts, sampling_params)
        
        # Save outputs and metadata
        output_data = {
            "outputs": outputs,
            "metadata": metadata
        }
        
        # Create pickle path
        pickle_path = get_pickle_path(args.output_dir, dataset_name)
        
        # Save to pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"Saved inference outputs for {dataset_name} to {pickle_path}")

def eval_mode(datasets_to_eval, args):
    """Run only the evaluation step using saved outputs"""
    all_results = {}
    
    for dataset_name in datasets_to_eval:
        pickle_path = get_pickle_path(args.output_dir, dataset_name)
        
        if not os.path.exists(pickle_path):
            print(f"No saved outputs found for {dataset_name} at {pickle_path}. Skipping evaluation.")
            continue
        
        print(f"Loading saved outputs for {dataset_name} from {pickle_path}")
        
        # Load the saved outputs and metadata
        with open(pickle_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        outputs = saved_data["outputs"]
        metadata = saved_data["metadata"]
        
        # Process outputs
        results = process_outputs(outputs, metadata, args.eval_threads)
        all_results[dataset_name] = results
        
        metrics = calculate_metrics(results)
        
        output_dict = {
            "results": results,
            "metrics": metrics,
            "config": vars(args)
        }
        
        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
        print(f"{dataset_name.upper()} Results:")
        print(f"  Total samples: {len(results)}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'sub_accuracies' in metrics:
            print("  Task/Category Accuracies:")
            for task, acc in metrics['sub_accuracies'].items():
                print(f"    {task}: {acc:.4f}")
        print()
    
    print(f"All evaluation results saved to {args.output_dir}")

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to evaluate
    datasets_to_eval = args.datasets.split(",") if args.datasets != "all" else [
        "geo3k", "wemath", "mathvista", "mathverse", "mathvision", "dynamath"
    ]
    
    # Dictionary to store all samples
    all_samples = {}
    
    # Load datasets only if needed (inference or all mode)
    if args.mode in ["inference", "all"]:
        # Load datasets based on selection
        for dataset_name in datasets_to_eval:
            if dataset_name == "geo3k":
                all_samples["geo3k"] = load_geo3k_dataset(args.data_path)
                print(f"Loaded {len(all_samples['geo3k'])} samples from Geo3K")
            
            elif dataset_name == "logicvista":
                all_samples["logicvista"] = load_logicvista_dataset(args.data_path)
                print(f"Loaded {len(all_samples['logicvista'])} samples from LogicVista")
            
            elif dataset_name == "wemath":
                all_samples["wemath"] = load_wemath_dataset(args.data_path)
                print(f"Loaded {len(all_samples['wemath'])} samples from WeMath")
            
            elif dataset_name == "mathvista":
                all_samples["mathvista"] = load_mathvista_dataset(args.data_path)
                print(f"Loaded {len(all_samples['mathvista'])} samples from MathVista")
            
            elif dataset_name == "mathverse":
                all_samples["mathverse"] = load_mathverse_dataset(args.data_path)
                print(f"Loaded {len(all_samples['mathverse'])} samples from MathVerse")
                
            elif dataset_name == "dynamath":
                all_samples["dynamath"] = load_dynamath_dataset(args.tsv_data_path, args.data_path)
                print(f"Loaded {len(all_samples['dynamath'])} samples from Dynamath")
                
            elif dataset_name == "olympiadbench":
                all_samples["olympiadbench"] = load_olympiadbench_dataset(args.data_path)
                print(f"Loaded {len(all_samples['olympiadbench'])} samples from OlympiadBench")
            
            elif dataset_name == "mathvision":
                all_samples["mathvision"] = load_mathvision_dataset(args.tsv_data_path, args.data_path)
                print(f"Loaded {len(all_samples['mathvision'])} samples from MathVision")
            
            elif dataset_name == "hallubench":
                all_samples["hallubench"] = load_hallubench_dataset(args.data_path)
                print(f"Loaded {len(all_samples['hallubench'])} samples from HalluBench")
        
        if not all_samples:
            print("No datasets loaded. Please check the paths and dataset names.")
            return
    
    # Execute based on the specified mode
    if args.mode == "inference":
        inference_mode(all_samples, args)
    
    elif args.mode == "eval":
        eval_mode(datasets_to_eval, args)
    
    elif args.mode == "all":
        # Run inference first
        inference_mode(all_samples, args)
        
        # Then run evaluation
        eval_mode(datasets_to_eval, args)

if __name__ == "__main__":
    main()