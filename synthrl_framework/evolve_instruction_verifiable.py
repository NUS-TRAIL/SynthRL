import argparse
import json
import numpy as np
import gc
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from datasets import Dataset
import io

import random
import logging
import os
import PIL.Image
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai

from verify_utils import (
    SYSTEM_MESSAGES,
    call_gemini_api,
    call_qwen_api,
    get_evolve_prompt,
    extract_evolved_question,
    judge_evolved_question_quality,
    get_solver_prompt,
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
qwen_api_key = os.getenv("SILICON_FLOW_API_KEY")

def solver_rollout(
    image,
    question: str,
    correct_answer: str,
    rollout_count: int,
    model_name: str = "gemini-1.5-flash-002",
    solver_name: str = None,
    verbose: bool = False
) -> Tuple[List[str], int, float]:  # Return type includes avg_reasoning_steps
    """
    Generate multiple rollouts to solve the question and check how many pass.

    Args:
        image: PIL Image for the question
        question: The question text
        correct_answer: The correct answer
        rollout_count: Number of rollouts to perform
        model_name: Model to use for verification
        solver_name: Model to use for solving (defaults to model_name if None)
        verbose: Whether to log detailed information

    Returns:
        Tuple of (list of responses, count of correct answers, average reasoning steps for correct responses)
    """
    responses = []
    correct_count = 0
    total_reasoning_steps = 0
    correct_response_count = 0

    # Use solver_name if provided, otherwise fall back to model_name
    solver_model = solver_name if solver_name else model_name

    # System message for the solver
    system_instruction = SYSTEM_MESSAGES['solver']

    for i in range(rollout_count):
        try:
            # Generate solver prompt
            solver_prompt = get_solver_prompt(question)
            
            # Include system instruction before solver prompt
            combined_prompt = f"\n{system_instruction} {solver_prompt}"
            solver_content = [image, combined_prompt]

            # Choose which API to call based on the solver model name
            if "gemini" in solver_model.lower():
                solving_response = call_gemini_api(
                    content=solver_content,
                    model_name=solver_model,
                    temperature=1.0,  # Use higher temperature for diversity in rollouts
                    max_tokens=2048,
                    retries=2  # Increased retries for better reliability
                )
            elif "qwen" in solver_model.lower():

                solving_response = call_qwen_api(
                    content=solver_content,
                    model_name=solver_model,
                    temperature=1.0,
                    max_tokens=2048,
                    retries=2
                )
            else:
                raise ValueError(f"Unsupported model family for solver: {solver_model}")

            # Extract the answer from solver's response
            generated_answer = extract_final_answer(solving_response)

            # Store the full response
            responses.append(solving_response)

            # Verify if the answer matches the correct answer
            is_correct = verify_answers_match(correct_answer, generated_answer, "gemini-2.0-flash")
            if is_correct:
                correct_count += 1

                # Count reasoning steps for correct responses
                # Simply counting sentences by splitting on periods, question marks, and exclamation points
                sentences = [s.strip() for s in solving_response.split('\n') if s.strip()]
                num_steps = len(sentences)
                total_reasoning_steps += num_steps
                correct_response_count += 1

            if verbose:
                logging.info(f"Rollout {i+1}/{rollout_count}: {'✓' if is_correct else '✗'}")

        except Exception as e:
            logging.error(f"Error in rollout {i+1}: {e}")
            responses.append(f"Error: {str(e)}")

    # Calculate average reasoning steps for correct responses
    avg_reasoning_steps = 0
    if correct_response_count > 0:
        avg_reasoning_steps = total_reasoning_steps / correct_response_count

    return responses, correct_count, avg_reasoning_steps


# MODIFIED function: evolve_question_with_same_answer
def evolve_question_with_same_answer(
    sample: Dict,
    model_name: str = "gemini-1.5-flash-002",
    solver_name: str = None,
    judge_threshold: int = 7,
    max_attempts: int = 5,
    rollout: int = 8,
    difficulty_level: int = 0,
    verbose: bool = False
) -> Tuple[List[Dict], Dict, Optional[Dict]]:
    """
    Evolves the question in a sample while maintaining the same answer,
    focusing on finding a 'harder' question (lower pass rate).

    Uses the following flow:
    1. Evolve the question
    2. Judge the quality of the evolved question
    3. If quality meets threshold, use solver to generate answer with rollouts
    4. Verify if solver answers match original answer and check difficulty (hardness)

    Returns:
        Tuple of (list of all attempted evolved questions, stats dict, best harder question dict or None)
    """
    # Extract question, answer, and original pass rate
    question = sample['question']
    answer = sample['answer']
    image = sample['image']  # This is now directly the PIL Image
    image_id = sample['id']
    original_pass = sample.get('original_pass', 0)  # Get the original pass count
    original_reasoning_steps = sample.get('original_reasoning_steps', 0)

    # Use solver_name if provided, otherwise use model_name
    solver_model = solver_name if solver_name else model_name

    # Initialize tracking variables
    questions = [{'id': image_id, 'question': question, 'answer': answer, 'score': None, 'image': image}]
    current_question = question

    # Initialize variable to track the best harder question found
    best_harder_question = None # Tracks the best question that's harder

    # Statistics tracking
    stats = {
        'successful_evolutions': 0,
        'failed_evolutions': 0,
        'quality_failures': 0,
        'solver_failures': 0,
        'api_errors': 0,
        'extraction_errors': 0,
        'skipped_due_to_difficulty': 0,
        'total_evolved_questions_sent_to_solver': 0  # NEW: Track how many were sent to solver
    }

    # Check if this sample is eligible for processing
    if original_pass < difficulty_level:
        logging.info(f"Sample {image_id} skipped: original_pass ({original_pass}) < difficulty_level ({difficulty_level})")
        stats['skipped_due_to_difficulty'] = 1
        return questions, stats, None

    # Evolution loop
    for attempt in range(max_attempts):
        if verbose:
            logging.info(f"Attempt {attempt+1}/{max_attempts} for image ID {image_id}")

        # STEP 1: EVOLVE the question
        system_instruction = SYSTEM_MESSAGES['evolve']
        evolve_prompt = get_evolve_prompt(current_question)
        content = [image, "\n", evolve_prompt]

        try:
            # Call API to evolve the question
            assistant_reply = call_gemini_api(
                content=content,
                model_name=model_name,
                temperature=1.0,
                max_tokens=24576,
                retries=2,
                system_instruction=system_instruction
            )

            if verbose:
                logging.info(f"API response: {assistant_reply}")

            # Extract the evolved question
            new_question = extract_evolved_question(assistant_reply)
            if not new_question:
                logging.warning(f"Failed to extract evolved question at attempt {attempt+1} for image ID {image_id}")
                stats['extraction_errors'] += 1
                continue

            # STEP 2: JUDGE the quality of the evolved question
            quality_result = judge_evolved_question_quality(
                original_question=current_question,
                evolved_question=new_question,
                image=image,
                model_name="gemini-2.0-flash"
            )

            quality_score = quality_result.get('score', 0)
            quality_explanation = quality_result.get('explanation', '')

            if verbose:
                logging.info(f"Quality score: {quality_score}")
                logging.info(f"Quality explanation: {quality_explanation[:100]}...")

            # Create a result for the evolved question with quality information
            evolved_result = {
                'id': image_id,
                'question': new_question,
                'answer': answer,
                'quality_score': quality_score,
                'quality_explanation': quality_explanation,
                'attempt': attempt + 1,
                'passed_quality_check': quality_score >= judge_threshold,
                'image': image,
                'solver_responses': None,  # Will be populated if sent to solver
                'correct_count': None,     # Will be populated if sent to solver
                'avg_reasoning_steps': None,  # Will be populated if sent to solver
                'is_harder': None          # Will be populated if sent to solver
            }
            
            # Only proceed with solver if quality meets threshold
            if quality_score >= judge_threshold:
                # STEP 3: SOLVE the evolved question with rollouts
                stats['total_evolved_questions_sent_to_solver'] += 1  # NEW: Count questions sent to solver
                
                solver_responses, correct_count, avg_reasoning_steps = solver_rollout(
                    image=image,
                    question=new_question,
                    correct_answer=answer,
                    rollout_count=rollout,
                    model_name=model_name,  # For verification
                    solver_name=solver_model,  # For solving
                    verbose=verbose
                )

                if verbose:
                    logging.info(f"Rollout results: {correct_count}/{rollout} correct, avg steps: {avg_reasoning_steps:.2f}")
                    logging.info(f"Reasoning steps comparison: Evolved {avg_reasoning_steps:.2f} vs Original {original_reasoning_steps:.2f}")

                # MODIFIED: Update the result with solver information
                evolved_result.update({
                    'solver_responses': solver_responses,
                    'correct_count': correct_count,
                    'avg_reasoning_steps': avg_reasoning_steps,
                    'pass_rate': f"{correct_count}/{rollout}"  # NEW: Record as string for readability
                })

                # Check if the evolved question is harder
                # Condition: More than 1 success (not impossible) and fewer successes than original (harder)
                is_harder = correct_count >= 4 and correct_count < original_pass - 1
                evolved_result['is_harder'] = is_harder  # Record whether it meets the criteria

                if is_harder:
                    # This is a valid evolution that meets our criteria
                    current_question = new_question  # Evolve from this harder question
                    stats['successful_evolutions'] += 1

                    # Update the best harder question found so far
                    # Lower correct_count means harder, which is better
                    if (best_harder_question is None or
                        correct_count < best_harder_question['correct_count']):
                        best_harder_question = evolved_result
                        if verbose:
                             logging.info(f"Found new best harder question (Pass: {correct_count}/{rollout}): {new_question}")
                else:
                    # Log why it wasn't considered a successful 'harder' evolution
                    stats['solver_failures'] += 1
                    if correct_count <= 1:
                        logging.warning(f"Evolved question too hard or failed rollouts ({correct_count}/{rollout}) at attempt {attempt+1}")
                    elif correct_count >= original_pass - 1:
                         logging.warning(f"Evolved question not harder than original: {correct_count}/{rollout} correct vs {original_pass} original")
            else: # Quality score below threshold
                stats['quality_failures'] += 1
                logging.warning(f"Quality score ({quality_score}) below threshold ({judge_threshold}) at attempt {attempt+1}")
            
            # MODIFIED: Add the evolved question to our list regardless of whether it meets criteria
            questions.append(evolved_result)

        except Exception as e:
            logging.error(f"Error during evolution attempt {attempt+1} for image ID {image_id}: {e}")
            stats['api_errors'] += 1
            # Continue to next attempt

    # If no successful evolution happened after all attempts
    if stats['successful_evolutions'] == 0:
        stats['failed_evolutions'] += 1

    # Return all attempted questions, stats, and the best harder one found (or None)
    return questions, stats, best_harder_question

def process_sample(sample_dict, args):
    """
    Process a single sample to evolve its question while maintaining the same answer,
    aiming for a harder question.
    """
    try:
        # Extract required fields
        sample = {
            'id': sample_dict['id'],
            'image': sample_dict.get('image', sample_dict.get('images', None)),
            'question': sample_dict.get('problem', sample_dict.get('original_question', '')),
            'answer': sample_dict['answer'],
            'original_pass': sample_dict.get('original_pass', 0),
            'original_reasoning_steps': sample_dict.get('original_reasoning_steps', 0),
            'original_problem_responses': sample_dict.get('original_problem_responses', [])
        }


        # Convert bytes to PIL Image if needed
        if isinstance(sample['image'], dict) and 'bytes' in sample['image']:
            sample['image'] = PIL.Image.open(io.BytesIO(sample['image']['bytes']))

        # Evolve the question - expecting only 3 return values now
        evolved_questions, stats, harder_question = evolve_question_with_same_answer(
            sample,
            model_name=args.model_name,
            solver_name=args.solver_name,
            judge_threshold=args.judge_threshold,
            max_attempts=args.max_attempts,
            rollout=args.rollout,
            difficulty_level=args.difficulty_level,
            verbose=args.verbose
        )

        # Result with image preserved as PIL Image
        # REMOVED fields related to 'more_steps'
        result = {
            'id': sample['id'],
            'image': sample['image'],  # Preserve the image as PIL Image
            'original_question': sample['question'],
            'original_answer': sample['answer'],
            'original_pass': sample['original_pass'],
            'original_reasoning_steps': sample['original_reasoning_steps'],
            'original_problem_responses': sample['original_problem_responses'],
            'evolved_questions': evolved_questions if evolved_questions else [], # Keep track of all attempts
            'evolved_question': harder_question['question'] if harder_question else None,
            'evolved_pass': harder_question['correct_count'] if harder_question else None,
            'evolved_reasoning_steps': harder_question['avg_reasoning_steps'] if harder_question else None,
            'evolved_problem_responses': harder_question['solver_responses'] if harder_question else [],
            'stats': stats # Include the stats from the evolution process
        }
        return result
    except Exception as e:
        logging.error(f"Error processing sample {sample_dict.get('id', 'N/A')}: {str(e)}")
        # Optionally return stats even on failure
        return {'id': sample_dict.get('id', 'N/A'), 'stats': {'processing_error': 1}, 'evolved_question': None}
    
def main():
    parser = argparse.ArgumentParser(description='Evolve questions while preserving answers using Gemini API.')
    # --- (Keep all argument parsing as before) ---
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./evolved_questions', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='gemini-1.5-flash-002', help='Model to use for generating and judging')
    parser.add_argument('--solver_name', type=str, default=None, help='Model to use for rollout (defaults to model_name if not specified)')
    parser.add_argument('--judge_threshold', type=int, default=8, help='Minimum quality score (0-10) required for acceptance')
    parser.add_argument('--max_attempts', type=int, default=5, help='Maximum evolution attempts per question')
    parser.add_argument('--rollout', type=int, default=8, help='Number of rollouts for solver verification')
    parser.add_argument('--difficulty_level', type=int, default=4, help='Minimum original_pass needed to process sample')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--sample_limit', type=int, default=None, help='Maximum number of samples to process (None for all)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--remote_dataset', action='store_true', help='Load dataset from Hugging Face instead of disk')
    parser.add_argument('--images_in_list', action='store_true', help='Handle datasets where images are stored in lists')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use (train, validation, test)')
    args = parser.parse_args()

    # --- (Keep setup: solver_name logic, output dir, logging, seed, dataset loading, verification) ---
    # If solver_name not specified, use the same model as model_name
    if args.solver_name is None:
        args.solver_name = args.model_name
        logging.info(f"Using the same model for solver: {args.solver_name}")
    else:
        logging.info(f"Using different models for tasks: {args.model_name} (main) and {args.solver_name} (solver)")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/evolution.log")
        ]
    )
    random.seed(args.seed)
    logging.info(f"Configuration: judge_threshold={args.judge_threshold}, rollout={args.rollout}, difficulty_level={args.difficulty_level}")
    logging.info(f"Model: {args.model_name}, Max attempts: {args.max_attempts}")

    # Load dataset (simplified example, keep your original loading logic)
    logging.info(f"Loading dataset from {args.dataset_path}")
    try:
        from datasets import load_dataset, load_from_disk
        if args.remote_dataset:
             dataset = load_dataset(args.dataset_path, split=args.split)
        else:
             # Adapt path logic as needed
             try:
                 full_dataset_path = os.path.join('./sampled_data', args.dataset_path)
                 dataset = load_from_disk(full_dataset_path)
             except Exception:
                 dataset = load_from_disk(args.dataset_path) # Fallback
        logging.info(f"Loaded dataset with {len(dataset)} samples")
        if 'original_pass' not in dataset.column_names:
             logging.error("Dataset missing 'original_pass' field. Make sure to run rollout first!")
             return
        if args.sample_limit is not None and args.sample_limit < len(dataset):
             dataset = dataset.select(range(args.sample_limit))
             logging.info(f"Selected {len(dataset)} samples for processing")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Process dataset
    results = []
    # MODIFIED: Removed 'successful_evolutions_more_steps' from stats tracking
    total_stats = {
        'successful_evolutions': 0,
        'failed_evolutions': 0,
        'quality_failures': 0,
        'solver_failures': 0,
        'api_errors': 0,
        'extraction_errors': 0,
        'skipped_due_to_difficulty': 0,
        'processing_error': 0 # Add a counter for errors in process_sample
    }

    logging.info("Starting question evolution process")

    # Process in batches
    for i in range(0, len(dataset), args.batch_size):
        end_idx = min(i + args.batch_size, len(dataset))
        batch = dataset[i:end_idx]
        logging.info(f"Processing batch {i//args.batch_size + 1}/{(len(dataset) + args.batch_size - 1)//args.batch_size}")

        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            # --- (Keep the logic for preparing samples from different batch formats) ---
            # Example for dict of lists format:
            if isinstance(batch, dict) and 'id' in batch and isinstance(batch['id'], (list, tuple)):
                batch_size = len(batch['id'])
                for j in range(batch_size):
                    try:
                        # Prepare sample (extract image, question, answer, etc.)
                        # Simplified - use your original detailed preparation logic
                        image = batch.get('image', batch.get('images', [None]*batch_size))[j]
                        if args.images_in_list and isinstance(image, list) and len(image) > 0:
                            image = image[0]

                        prepared_sample = {
                            'id': batch['id'][j],
                            'image': image,
                            'problem': batch.get('problem', batch.get('original_question', ['']*batch_size))[j],
                            'answer': batch['answer'][j],
                            'original_pass': batch.get('original_pass', [0]*batch_size)[j],
                            'original_reasoning_steps': batch.get('original_reasoning_steps', [0]*batch_size)[j],
                            'original_problem_responses': batch.get('original_problem_responses', [[]]*batch_size)[j]
                        }
                        if not all(prepared_sample.get(k) is not None for k in ['image', 'problem', 'answer']):
                             logging.warning(f"Skipping sample {prepared_sample['id']} due to missing required fields")
                             continue

                        futures.append(executor.submit(
                            process_sample,
                            prepared_sample,
                            args
                        ))
                    except Exception as e:
                        logging.error(f"Error preparing sample index {j} for submission: {str(e)}")
                        total_stats['processing_error'] += 1 # Count preparation errors

            # --- (Add logic for other batch formats if necessary) ---
            else:
                 logging.error("Unsupported batch format encountered.")
                 # Handle list of dicts or other formats if needed

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batch"):
                try:
                    result = future.result(timeout=300)
                    if result:
                        batch_results.append(result)
                        # Update total stats - check if stats exist in result
                        if 'stats' in result:
                            for key in total_stats:
                                total_stats[key] += result['stats'].get(key, 0)
                        else:
                            # Handle cases where process_sample might have failed before returning stats
                            logging.warning(f"Result for sample {result.get('id', 'N/A')} missing 'stats' dictionary.")
                            total_stats['processing_error'] += 1


                except concurrent.futures.TimeoutError:
                    logging.error("A sample processing task timed out.")
                    total_stats['api_errors'] += 1 # Or a specific timeout error count
                except Exception as e:
                    logging.error(f"Error processing future result: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    total_stats['processing_error'] += 1 # Count errors during result retrieval


        results.extend(batch_results)

        # --- (Keep intermediate saving logic, ensure it handles the modified 'result' structure) ---
        # Example: Serialize and save batch results (adapt as needed)
        serializable_batch_results = []
        for res in batch_results:
            res_copy = res.copy()
            if 'image' in res_copy:
                res_copy['image'] = 'PIL Image object (removed for serialization)'
            if 'evolved_questions' in res_copy:
                res_copy['evolved_questions'] = [
                    {k: v if not isinstance(v, PIL.Image.Image) else 'PIL Image object'
                    for k, v in q.items()}
                    for q in res_copy.get('evolved_questions', [])
                ]
            # NEW: Add a summary of all evolved questions with their pass rates for easy analysis
            res_copy['evolved_questions_summary'] = [
                {
                    'attempt': q.get('attempt', 'unknown'),
                    'question': q.get('question', ''),
                    'quality_score': q.get('quality_score', None),
                    'correct_count': q.get('correct_count', None),
                    'pass_rate': q.get('pass_rate', None),
                    'is_harder': q.get('is_harder', None),
                    'passed_quality_check': q.get('passed_quality_check', None)
                }
                for q in res_copy.get('evolved_questions', [])[1:]  # Skip the original question
            ]
            serializable_batch_results.append(res_copy)
        try:
            with open(f"{args.output_dir}/results_batch_{i//args.batch_size}.json", 'w') as f:
                json.dump(serializable_batch_results, f, indent=2)
            logging.info(f"Saved batch {i//args.batch_size + 1} results")
        except Exception as e:
            logging.error(f"Failed to save intermediate batch results: {e}")


    # --- (Keep final saving logic for all_results.json, adapt serialization) ---
    serializable_results = []
    for result in results:
        result_copy = result.copy()
        if 'image' in result_copy:
            result_copy['image'] = 'PIL Image object (removed for serialization)'
        if 'evolved_questions' in result_copy:
             result_copy['evolved_questions'] = [
                 {k: v if not isinstance(v, PIL.Image.Image) else 'PIL Image object'
                  for k, v in q.items()}
                 for q in result_copy.get('evolved_questions', [])
             ]
        serializable_results.append(result_copy)
    with open(f"{args.output_dir}/all_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)


    # --- Calculate average quality score (no changes needed here) ---
    avg_quality_score = 0
    total_evolved_questions_judged = 0 # Count only those that were judged
    for result in results:
        # Iterate through evolved questions list (excluding original) from successful runs
        if result and 'evolved_questions' in result:
             # Start from index 1 to skip the original question placeholder
             for q in result['evolved_questions'][1:]:
                 if 'quality_score' in q and q['quality_score'] is not None: # Check quality score exists
                     avg_quality_score += q['quality_score']
                     total_evolved_questions_judged += 1

    if total_evolved_questions_judged > 0:
        avg_quality_score /= total_evolved_questions_judged


    # --- Save statistics - REMOVED 'successful_evolutions_more_steps' ---
    with open(f"{args.output_dir}/statistics.json", 'w') as f:
        json.dump({
            'total_samples_processed': len(results), # Samples for which processing was attempted
            'total_samples_in_dataset_slice': len(dataset), # Initial count before processing
            'total_evolved_questions_judged': total_evolved_questions_judged, # Count based on quality score presence
            'stats': total_stats, # Use the aggregated stats dictionary
            'average_quality_score': avg_quality_score,
            'configuration': {
                'judge_threshold': args.judge_threshold,
                'model_name': args.model_name,
                'solver_name': args.solver_name,
                'max_attempts': args.max_attempts,
                'rollout': args.rollout,
                'difficulty_level': args.difficulty_level
            }
        }, f, indent=2)

    # --- Create HuggingFace dataset - MODIFIED schema ---
    try:
        from datasets import Dataset as HFDataset

        # MODIFIED: Added fields to store all evolved questions
        dataset_dict = {
            'id': [],
            'image': [],
            'original_question': [],
            'original_answer': [],
            'original_pass': [],
            'original_reasoning_steps': [],
            'original_problem_responses': [],
            'evolved_question': [],  # Best harder question
            'evolved_quality_score': [],
            'evolved_pass': [],
            'evolved_reasoning_steps': [],
            'evolved_problem_responses': [],
            # NEW: Add fields for all questions that were sent to solver
            'all_evolved_questions': [],  # List of all evolved questions
            'all_quality_scores': [],     # List of quality scores
            'all_pass_rates': [],         # List of pass rates (e.g., "5/8")
            'all_is_harder': []           # List of whether each question is harder
        }

        # Populate the dataset
        for result in results:
            # NEW: Extract all evolved questions that were sent to solver
            all_evolved_questions = []
            all_quality_scores = []
            all_pass_rates = []
            all_is_harder = []
            
            for q in result.get('evolved_questions', [])[1:]:
                # Only include questions that were sent to solver (have solver_responses)
                if q.get('solver_responses') is not None:
                    all_evolved_questions.append(q.get('question', ''))
                    all_quality_scores.append(q.get('quality_score', None))
                    all_pass_rates.append(q.get('pass_rate', None))
                    all_is_harder.append(q.get('is_harder', None))
                    
            # Ensure result is valid and contains expected keys
            if not result or 'id' not in result:
                logging.warning("Skipping invalid result during HF dataset creation.")
                continue

            sample_id = result['id']
            image = result.get('image') # Use .get for safety
            # Ensure image is PIL object if not None
            if image and isinstance(image, dict) and 'bytes' in image:
                 try:
                     image = PIL.Image.open(io.BytesIO(image['bytes']))
                 except Exception as img_err:
                     logging.warning(f"Could not decode image for sample {sample_id}: {img_err}")
                     image = None # Set to None if decoding fails

            # Use .get with defaults for robustness
            original_question = result.get('original_question', '')
            original_answer = result.get('original_answer', '')
            original_pass = result.get('original_pass', 0)
            original_reasoning_steps = result.get('original_reasoning_steps', 0)
            original_problem_responses = result.get('original_problem_responses', [])

            evolved_question = result.get('evolved_question') # This is the selected 'harder' one
            evolved_pass = result.get('evolved_pass')
            evolved_reasoning_steps = result.get('evolved_reasoning_steps')
            evolved_problem_responses = result.get('evolved_problem_responses', [])

            # Find the quality score associated with the selected evolved_question
            evolved_quality_score = None
            if evolved_question and 'evolved_questions' in result:
                for q in result['evolved_questions']:
                    if q.get('question') == evolved_question:
                        evolved_quality_score = q.get('quality_score')
                        break # Found the score for the selected question

            # Append data
            dataset_dict['id'].append(sample_id)
            dataset_dict['image'].append(image)
            dataset_dict['original_question'].append(original_question)
            dataset_dict['original_answer'].append(original_answer)
            dataset_dict['original_pass'].append(original_pass)
            dataset_dict['original_reasoning_steps'].append(original_reasoning_steps)
            dataset_dict['original_problem_responses'].append(original_problem_responses)
            dataset_dict['evolved_question'].append(evolved_question)
            dataset_dict['evolved_quality_score'].append(evolved_quality_score)
            dataset_dict['evolved_pass'].append(evolved_pass)
            dataset_dict['evolved_reasoning_steps'].append(evolved_reasoning_steps)
            dataset_dict['evolved_problem_responses'].append(evolved_problem_responses)
            
            dataset_dict['all_evolved_questions'].append(all_evolved_questions)
            dataset_dict['all_quality_scores'].append(all_quality_scores)
            dataset_dict['all_pass_rates'].append(all_pass_rates)  
            dataset_dict['all_is_harder'].append(all_is_harder)


        # Create and save the dataset
        # Filter out potential None images before creating dataset if necessary
        # For simplicity, assuming HFDataset handles None images or they are filtered earlier
        hf_dataset = HFDataset.from_dict(dataset_dict)
        hf_dataset.save_to_disk(f"{args.output_dir}")
        logging.info(f"Saved HuggingFace dataset with {len(hf_dataset)} samples")
    except ImportError:
         logging.warning("datasets library not fully available. Skipping HuggingFace dataset creation.")
    except Exception as e:
        logging.error(f"Error creating HuggingFace dataset: {e}")
        import traceback
        logging.error(traceback.format_exc())


    # --- Print final statistics - REMOVED 'more_steps' references ---
    print("\n========== FINAL STATISTICS ==========")
    print(f"Total samples processed: {len(results)}")
    print(f"Samples skipped due to difficulty level: {total_stats['skipped_due_to_difficulty']}")
    # print(f"Total evolved questions judged: {total_evolved_questions_judged}") # Already in stats file
    print(f"Successful evolutions (found harder question): {total_stats['successful_evolutions']}")
    print(f"Failed evolutions (no harder question found): {total_stats['failed_evolutions']}")
    print(f"Quality failures: {total_stats['quality_failures']}")
    print(f"Solver failures (evolved question not harder): {total_stats['solver_failures']}")
    print(f"API errors: {total_stats['api_errors']}")
    print(f"Extraction errors: {total_stats['extraction_errors']}")
    print(f"Processing errors (sample prep/result handling): {total_stats['processing_error']}") # Added

    print("\n========== QUALITY METRICS ==========")
    print(f"Average quality score (of judged questions): {avg_quality_score:.2f}/10")
    print(f"Quality threshold used: {args.judge_threshold}")
    print(f"Difficulty level used: {args.difficulty_level}")
    print(f"Rollout count: {args.rollout}")

    print("======================================\n")

    logging.info("Question evolution process completed")
    logging.info(f"Final aggregated statistics: {total_stats}")
    logging.info(f"Results saved to {args.output_dir}")

    
if __name__ == "__main__":
    main()