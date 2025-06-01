import argparse
import json
import gc
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset

import random
import logging
import os
import PIL.Image
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
from openai import OpenAI

from mathruler.grader import extract_boxed_content, grade_answer

from verify_utils import (
    SYSTEM_MESSAGES,
    get_solver_prompt,
    call_gemini_api,
    call_qwen_api,
    extract_final_answer,
    verify_answers_match
)

# Configure APIs
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Initialize Qwen client
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
qwen_client = None
if dashscope_api_key:
    qwen_client = OpenAI(
        api_key=dashscope_api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    # print("======================================")
    # print(f"PREDICTED: {predict_str}")
    # print(f"GROUND TRUTH: {ground_truth}")
    # print(f"ANSWER: {answer}")
    # print("======================================")
    # print()
    return True if grade_answer(answer, ground_truth) else False, answer

def process_sample(sample, args):
    """Process a single sample from the dataset."""
    try:
        # Extract fields
        sample_id = sample['id']
        problem = sample['problem']
        correct_answer = sample['answer']
        
        # Handle image - check if it's in a list
        image = sample['images']
        if args.images_in_list and isinstance(image, list) and len(image) > 0:
            image = image[0]  # Take the first image from the list
        
        logging.info(f"Processing sample {sample_id}: {problem[:50]}...")
    
        # Generate n responses
        responses = []
        correct_count = 0
        error_count = 0
        api_error_count = 0
        total_reasoning_steps = 0
        correct_response_count = 0
        
        for i in range(args.rollout):
            try:
                # Generate solver prompt
                solver_prompt = get_solver_prompt(problem)
                system_instruction = SYSTEM_MESSAGES['solver']
                combined_prompt = f"\n{system_instruction} {solver_prompt}"
                solver_content = [image, combined_prompt]
                
                # Choose the appropriate API based on model name
                if "gemini" in args.model_name.lower():
                    solving_response = call_gemini_api(
                        content=solver_content,
                        model_name=args.model_name,
                        temperature=1.0,
                        max_tokens=2048,
                        retries=5
                    )
                elif "qwen" in args.model_name.lower():
                    # Pass the client as an argument to avoid recreating it
                    solving_response = call_qwen_api(
                        content=solver_content,
                        model_name=args.model_name,
                        temperature=1.0,
                        max_tokens=2048,
                        retries=5,
                        client=qwen_client  # Pass the initialized client
                    )
                else:
                    raise ValueError(f"Unsupported model family: {args.model_name}")
                
                # Extract the answer from solver's response
                generated_answer = extract_final_answer(solving_response)                
                # Store the full response
                responses.append(solving_response)
                
                # Verify if the answer matches the correct answer
                is_correct = verify_answers_match(correct_answer, generated_answer)
            
                if is_correct:
                    correct_count += 1
                    
                    # Count reasoning steps for correct responses
                    # Simply counting sentences by splitting on periods, question marks, and exclamation points
                    sentences = [s.strip() for s in solving_response.split('\n') if s.strip()]
                    num_steps = len(sentences)
                    total_reasoning_steps += num_steps
                    correct_response_count += 1
                
                logging.info(f"Sample {sample_id}, Rollout {i+1}/{args.rollout}: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                error_count += 1
                error_msg = str(e)
                if "API" in error_msg or "rate limit" in error_msg.lower() or "timeout" in error_msg.lower():
                    api_error_count += 1
                
                error_response = f"Error: {error_msg}"
                responses.append(error_response)
                logging.error(f"Error in rollout {i+1} for sample {sample_id}: {e}")
        
        # Calculate average reasoning steps for correct responses
        avg_reasoning_steps = 0
        if correct_response_count > 0:
            avg_reasoning_steps = total_reasoning_steps / correct_response_count
        
        # Prepare the processed sample
        processed_sample = {
            'id': sample_id,
            'images': sample['images'],
            'problem': problem,
            'answer': correct_answer,
            'original_problem_responses': responses,
            'original_pass': correct_count,
            'original_reasoning_steps': avg_reasoning_steps,
            'error_count': error_count,
            'api_error_count': api_error_count,
            'complete_responses': len(responses) - error_count
        }
        
        return processed_sample
        
    except Exception as e:
        logging.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        return {
            'id': sample.get('id', 'unknown'),
            'images': sample.get('images', None),
            'problem': sample.get('problem', ''),
            'answer': sample.get('answer', ''),
            'original_problem_responses': [f"Sample processing error: {str(e)}"],
            'original_pass': 0,
            'original_reasoning_steps': 0,
            'error_count': args.rollout,
            'api_error_count': 0,
            'complete_responses': 0
        }
    
def main():
    parser = argparse.ArgumentParser(description='Generate multiple responses for each question using AI APIs.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./sampled_data', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='gemini-1.5-flash-002', help='Model to use (gemini or qwen)')
    parser.add_argument('--rollout', type=int, default=8, help='Rollouts per question')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--sample_limit', type=int, default=None, help='Maximum number of samples to process (None for all)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--remote_dataset', action='store_true', help='Load dataset from Hugging Face instead of disk')
    parser.add_argument('--images_in_list', action='store_true', help='Handle datasets where images are stored in lists')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use (train, validation, test)')
    args = parser.parse_args()
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/rollout_process.log")
        ]
    )
    
    # Set random seed
    random.seed(args.seed)
    
    # Log configuration
    logging.info(f"Configuration: model={args.model_name}, rollout={args.rollout}")
    
    # Load dataset
    logging.info(f"Loading dataset from {args.dataset_path}")
    try:
        from datasets import load_dataset, load_from_disk
        
        # Determine if we should load from HuggingFace or disk
        if args.remote_dataset:
            try:
                # Explicitly load the specified split (defaults to 'train')
                dataset = load_dataset(args.dataset_path, split=args.split)
                logging.info(f"Loaded HuggingFace dataset {args.dataset_path} (split: {args.split}) with {len(dataset)} samples")
            except Exception as e:
                logging.error(f"Failed to load dataset from HuggingFace: {e}")
                return
        else:
            # Try loading from disk with different path options
            try:
                # First try with the sampled_data prefix
                full_dataset_path = os.path.join('./sampled_data', args.dataset_path)
                logging.info(f"Looking for dataset at {full_dataset_path}")
                dataset = load_from_disk(full_dataset_path)
            except Exception as e1:
                logging.warning(f"Could not load from {full_dataset_path}: {e1}")
                try:
                    # Then try direct path
                    dataset = load_from_disk(args.dataset_path)
                    logging.info(f"Loaded dataset from direct path with {len(dataset)} samples")
                except Exception as e2:
                    logging.error(f"Failed to load dataset from disk: {e2}")
                    return
        
        logging.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    # Sample dataset if needed
    if args.sample_limit is not None and args.sample_limit < len(dataset):
        dataset = dataset.select(range(args.sample_limit))
        logging.info(f"Selected {len(dataset)} samples for processing")

    # Process all samples
    all_processed_samples = []
    
    # Statistics counters
    total_questions = 0
    total_responses = 0
    total_errors = 0
    total_api_errors = 0
    questions_with_incomplete_rollouts = 0
    total_avg_reasoning_steps = 0
    
    # Process in batches to manage memory
    for i in range(0, len(dataset), args.batch_size):
        end_idx = min(i + args.batch_size, len(dataset))
        batch = dataset[i:end_idx]
        logging.info(f"Processing batch {i//args.batch_size + 1}/{(len(dataset) + args.batch_size - 1)//args.batch_size}")
        
        # Process samples in parallel
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            
            # Check the format of the batch
            if isinstance(batch, dict) and 'id' in batch and isinstance(batch['id'], (list, tuple)):
                # Batch is a dictionary of lists (common with HuggingFace datasets)
                batch_size = len(batch['id'])
                for j in range(batch_size):
                    sample = {k: batch[k][j] for k in batch}
                    futures.append(executor.submit(process_sample, sample, args))
            else:
                # Batch is a list of dictionaries
                for sample in batch:
                    futures.append(executor.submit(process_sample, sample, args))
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
                processed_sample = future.result()
                if processed_sample:
                    # Update statistics
                    total_questions += 1
                    total_responses += processed_sample['complete_responses']
                    total_errors += processed_sample['error_count']
                    total_api_errors += processed_sample['api_error_count']
                    total_avg_reasoning_steps += processed_sample['original_reasoning_steps']
                    
                    if processed_sample['complete_responses'] < args.rollout:
                        questions_with_incomplete_rollouts += 1
                        
                    batch_results.append(processed_sample)
        
        all_processed_samples.extend(batch_results)
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Create a new dataset from the processed samples
    output_dataset = Dataset.from_list(all_processed_samples)
    
    # Generate output path
    dataset_basename = os.path.basename(args.dataset_path)
    
    # Include the model type and size in the output path
    if "qwen" in args.model_name.lower():
        # Extract model size for Qwen models
        if "7b" in args.model_name.lower():
            model_identifier = "qwen7b_v2"
        elif "32b" in args.model_name.lower():
            model_identifier = "qwen32b"
        elif "72b" in args.model_name.lower():
            model_identifier = "qwen72b"
        else:
            # Default if size can't be determined
            model_identifier = "qwen"
    elif "gemini" in args.model_name.lower():
        # For Gemini, just use "gemini"
        model_identifier = "gemini"
    else:
        # Generic fallback
        model_identifier = "model"
        
    output_path = os.path.join(args.output_dir, f"{dataset_basename}_rollout{args.rollout}_{model_identifier}")
    
    # Save the dataset
    output_dataset.save_to_disk(output_path)
    logging.info(f"Saved processed dataset with {len(output_dataset)} samples to {output_path}")
    
    # Print statistics
    print("\n===== ROLLOUT STATISTICS =====")
    print(f"Total questions processed: {total_questions}")
    print(f"Expected total responses: {total_questions * args.rollout}")
    print(f"Successful responses: {total_responses}")
    print(f"Total errors: {total_errors} ({(total_errors/(total_questions * args.rollout) * 100):.2f}%)")
    print(f"API-related errors: {total_api_errors} ({(total_api_errors/max(1, total_errors) * 100):.2f}% of errors)")
    print(f"Questions with incomplete rollouts: {questions_with_incomplete_rollouts} ({(questions_with_incomplete_rollouts/total_questions * 100):.2f}%)")
    print(f"Average responses per question: {total_responses/max(1, total_questions):.2f} out of {args.rollout}")
    print(f"Average reasoning steps for correct responses: {total_avg_reasoning_steps/max(1, total_questions):.2f}")
    print("==============================\n")
    
    # Also log statistics
    logging.info("\n===== ROLLOUT STATISTICS =====")
    logging.info(f"Total questions processed: {total_questions}")
    logging.info(f"Expected total responses: {total_questions * args.rollout}")
    logging.info(f"Successful responses: {total_responses}")
    logging.info(f"Total errors: {total_errors} ({(total_errors/(total_questions * args.rollout) * 100):.2f}%)")
    logging.info(f"API-related errors: {total_api_errors} ({(total_api_errors/max(1, total_errors) * 100):.2f}% of errors)")
    logging.info(f"Questions with incomplete rollouts: {questions_with_incomplete_rollouts} ({(questions_with_incomplete_rollouts/total_questions * 100):.2f}%)")
    logging.info(f"Average responses per question: {total_responses/max(1, total_questions):.2f} out of {args.rollout}")
    logging.info(f"Average reasoning steps for correct responses: {total_avg_reasoning_steps/max(1, total_questions):.2f}")
    logging.info("==============================\n")


if __name__ == "__main__":
    main()