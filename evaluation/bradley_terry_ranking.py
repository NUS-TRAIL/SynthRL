import os
import sys
import io
import time
import logging
import random
import numpy as np
import pandas as pd
from PIL import Image
import PIL
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import google.generativeai as genai
import argparse
from dotenv import load_dotenv
from utils.processing import load_image
import json
import concurrent.futures

# Configure APIs
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Import data loaders from utils
from utils.data_loaders import (
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_dynamath_dataset,
    load_mathvision_dataset,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def call_gemini_api(content, model_name="gemini-2.0-flash", temperature=0.5, max_tokens=512, retries=2, system_instruction=None):
    """
    Modified to handle PIL Images properly for Gemini API
    """
    model = genai.GenerativeModel(model_name=model_name)
    
    if system_instruction:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
    
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # Convert content list to proper format
    formatted_content = []
    for item in content:
        if isinstance(item, PIL.Image.Image):
            # Convert PIL Image to bytes
            with io.BytesIO() as bio:
                item.save(bio, format='PNG')
                img_bytes = bio.getvalue()
            formatted_content.append({
                "mime_type": "image/png",
                "data": img_bytes
            })
        else:
            formatted_content.append(item)

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                formatted_content,
                generation_config=generation_config
            )
            return response.text

        except Exception as e:
            logging.error(f"API call error (attempt {attempt + 1}/{retries + 1}): {str(e)}")
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise

def compare_samples(sample1, sample2, args):
    """Compare two samples using Gemini API to determine which is more difficult"""
    system_instruction = """
    Compare math problems based on their difficulty. Consider reasoning steps, domain knowledge needed, 
    and computational complexity in your assessment.
    """
    
    # Load images
    image1 = load_image(sample1['image_path'], args.min_pixels, args.max_pixels)
    image2 = load_image(sample2['image_path'], args.min_pixels, args.max_pixels)
    
    if image1 is None or image2 is None:
        return "tie"  # Default to tie if either image can't be loaded
    
    # Create a comparison prompt that doesn't overlap with system instruction
    comparison_prompt = """
    Which of these two math problems is more difficult?
    
    Provide a brief explanation comparing their difficulty levels, then end with exactly one of:
    "WINNER: FIRST", "WINNER: SECOND", or "WINNER: TIE"
    """
    
    # Randomly determine the order to prevent position bias
    randomize_order = random.choice([True, False])
    
    if randomize_order:
        first_sample, first_image = sample1, image1
        second_sample, second_image = sample2, image2
        winner_map = {"WINNER: FIRST": "sample1", "WINNER: SECOND": "sample2", "WINNER: TIE": "tie"}
    else:
        first_sample, first_image = sample2, image2
        second_sample, second_image = sample1, image1
        winner_map = {"WINNER: FIRST": "sample2", "WINNER: SECOND": "sample1", "WINNER: TIE": "tie"}
    
    # Combine the images with the comparison prompt
    combined_content = [
        first_image,
        "FIRST PROBLEM:",
        first_sample['question'],  # Note: The data still uses 'question' as the key
        second_image,
        "SECOND PROBLEM:",
        second_sample['question'],  # Note: The data still uses 'question' as the key
        comparison_prompt
    ]
    
    # Call the API
    response = call_gemini_api(
        combined_content, 
        system_instruction=system_instruction,
        temperature=0.6
    )
    
    # Parse the response for the winner
    if "WINNER: FIRST" in response:
        return winner_map["WINNER: FIRST"]
    elif "WINNER: SECOND" in response:
        return winner_map["WINNER: SECOND"]
    else:
        return "tie"

def conduct_battles(all_samples, opponents_per_sample=32, args=None):
    all_battles = {}
    num_workers = getattr(args, 'num_workers', None) or os.cpu_count()

    for dataset_name, samples in all_samples.items():
        battles = []
        logging.info(f"Conducting unique battles within dataset: {dataset_name} ({len(samples)} samples)") # Modified log slightly

        battle_tasks = []

        for i, sample1 in enumerate(samples):
            potential_opponents = samples[:i] + samples[i+1:]

            if not potential_opponents:
                 logging.warning(f"Sample {sample1.get('id', i)} has no potential opponents.")
                 continue

            num_to_select = min(opponents_per_sample, len(potential_opponents))

            if num_to_select > 0:
                try:
                    opponents = random.sample(potential_opponents, k=num_to_select)
                except ValueError as e:
                    logging.error(f"Error sampling opponents for sample {sample1.get('id', 'N/A')}: {e}. Skipping.")
                    continue # Skip this sample1 if sampling fails

                for sample2 in opponents:
                    battle_tasks.append((sample1, sample2))
            else:
                 logging.warning(f"Sample {sample1.get('id', 'N/A')} has 0 potential opponents after check. Skipping.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_battle = {
                executor.submit(compare_samples, sample1, sample2, args): (sample1, sample2)
                for sample1, sample2 in battle_tasks
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_battle),
                              total=len(future_to_battle),
                              desc=f"Processing {dataset_name}"):
                sample1, sample2 = future_to_battle[future]
                try:
                    winner = future.result()
                    battle_record = {
                        "sample_a_id": sample1['id'],
                        "sample_b_id": sample2['id'],
                        "winner": "sample_a" if winner == "sample1" else "sample_b" if winner == "sample2" else "tie"
                    }
                    battles.append(battle_record)
                except Exception as e:
                    logging.error(f"Error in battle between {sample1.get('id', 'N/A')} and {sample2.get('id', 'N/A')}: {str(e)}")

        if battles:
            all_battles[dataset_name] = pd.DataFrame(battles)
        else:
            logging.warning(f"No battles recorded for dataset: {dataset_name}")
            all_battles[dataset_name] = pd.DataFrame(columns=["sample_a_id", "sample_b_id", "winner"])

    return all_battles

def compute_bt_ratings(battles_df, scale=400, base=10, init_rating=1000):
    """
    Compute Bradley-Terry ratings from battle results.
    
    Args:
        battles_df: DataFrame with columns [sample_a_id, sample_b_id, winner]
        scale: Rating scale parameter (400 for traditional Elo)
        base: Base of the exponential (10 for traditional Elo)
        init_rating: Initial rating value
        
    Returns:
        Series with ratings for each sample
    """
    # Create pivot tables for different outcomes
    # Count battles where sample_a won
    ptbl_a_win = pd.pivot_table(
        battles_df[battles_df["winner"] == "sample_a"],
        index="sample_a_id",
        columns="sample_b_id",
        aggfunc="size",
        fill_value=0,
    )
    
    # Count ties
    if sum(battles_df["winner"] == "tie") == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            battles_df[battles_df["winner"] == "tie"],
            index="sample_a_id",
            columns="sample_b_id",
            aggfunc="size",
            fill_value=0,
        )
        # Add transpose because ties are symmetric
        ptbl_tie = ptbl_tie + ptbl_tie.T
    
    # Count battles where sample_b won
    ptbl_b_win = pd.pivot_table(
        battles_df[battles_df["winner"] == "sample_b"],
        index="sample_a_id",
        columns="sample_b_id",
        aggfunc="size",
        fill_value=0,
    )
    
    # Combine all battle outcomes with weights
    # Wins count as 2, ties count as 1 to each side
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie
    
    # Map samples to indices
    samples = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)
    
    # Prepare data for logistic regression
    p = len(samples)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)
    
    cur_row = 0
    sample_weights = []
    for sample_a in ptbl_win.index:
        for sample_b in ptbl_win.columns:
            if sample_a == sample_b:
                continue
                
            # Skip if no battles were conducted
            if np.isnan(ptbl_win.loc[sample_a, sample_b]) or np.isnan(ptbl_win.loc[sample_b, sample_a]):
                continue
                
            # Add row for sample_a winning
            X[cur_row, samples[sample_a]] = +np.log(base)
            X[cur_row, samples[sample_b]] = -np.log(base)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[sample_a, sample_b])
            
            # Add row for sample_b winning
            X[cur_row + 1, samples[sample_a]] = np.log(base)
            X[cur_row + 1, samples[sample_b]] = -np.log(base)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[sample_b, sample_a])
            cur_row += 2
            
    # Trim to actual used rows
    X = X[:cur_row]
    Y = Y[:cur_row]
    
    if cur_row == 0:
        logging.error("No valid battles to compute ratings")
        return pd.Series()
    
    # Fit logistic regression model
    # No penalty or intercept for pure Bradley-Terry
    lr = LogisticRegression(fit_intercept=False, penalty='l2', C=1e9, max_iter=100, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    
    # Convert coefficients to ratings
    bt_ratings = scale * lr.coef_[0] + init_rating
    ratings = pd.Series(bt_ratings, index=samples.index).sort_values(ascending=False)
    
    return ratings

def compute_bootstrap_sample(battles_df, seed):
    """
    Compute ratings for a single bootstrap sample.
    
    Args:
        battles_df: DataFrame with battle results
        seed: Random seed for reproducibility
        
    Returns:
        Series with ratings
    """
    np.random.seed(seed)
    # Sample with replacement from battles
    bootstrap_sample = battles_df.sample(frac=1.0, replace=True)
    # Compute ratings on bootstrap sample
    return compute_bt_ratings(bootstrap_sample)

def compute_bootstrap_confidence_intervals(battles_df, bootstrap_rounds=100, num_workers=16):
    """
    Compute confidence intervals using bootstrap resampling with parallelization.
    
    Args:
        battles_df: DataFrame with battle results
        bootstrap_rounds: Number of bootstrap rounds
        num_workers: Number of worker processes to use
        
    Returns:
        DataFrame with ratings and confidence intervals
    """
    # Use ProcessPoolExecutor for CPU-bound task
    bootstrap_results = []
    
    if num_workers > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all bootstrap jobs with different seeds
                futures = [executor.submit(compute_bootstrap_sample, battles_df, i) 
                          for i in range(bootstrap_rounds)]
                
                # Collect results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=bootstrap_rounds, 
                                  desc="Bootstrap"):
                    try:
                        ratings = future.result()
                        bootstrap_results.append(ratings)
                    except Exception as e:
                        logging.error(f"Error in bootstrap sample: {e}")
        except Exception as e:
            logging.error(f"Error in parallel bootstrap processing: {e}")
            logging.info("Falling back to sequential processing")
            num_workers = 1
    
    # Fall back to sequential processing if parallel processing failed or wasn't requested
    if num_workers <= 1 or len(bootstrap_results) == 0:
        bootstrap_results = []
        for i in tqdm(range(bootstrap_rounds), desc="Bootstrap"):
            try:
                ratings = compute_bootstrap_sample(battles_df, i)
                bootstrap_results.append(ratings)
            except Exception as e:
                logging.error(f"Error in bootstrap sample {i}: {e}")
    
    # Combine bootstrap results
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    # Some samples might not appear in all bootstrap samples
    # Fill NaN values with the minimum rating to be conservative
    min_rating = bootstrap_df.min().min()
    bootstrap_df = bootstrap_df.fillna(min_rating)
    
    # Compute statistics from bootstrap samples
    result = pd.DataFrame({
        'rating': bootstrap_df.median(),
        'lower_ci': bootstrap_df.quantile(0.025),
        'upper_ci': bootstrap_df.quantile(0.975)
    }).sort_values('rating', ascending=False)
    
    return result

def main():
    """Main function to run the battle and rating process."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Battle samples and compute quality ratings")
    parser.add_argument("--data_path", type=str, default='./data', help="Base directory containing dataset files")
    parser.add_argument("--tsv_data_path", type=str, default='~/LMUData', help="Path to TSV data")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["mathvision"],
                        # default=["wemath", "mathvista", "dynamath", "mathvision", "mathverse"],
                        help="List of datasets to process")
    parser.add_argument("--bootstrap_rounds", type=int, default=32, help="Number of bootstrap rounds for CI")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for API calls")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for API response")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--use_standard_variants", action="store_false", 
                    dest="use_filtered_variants",
                    help="Use standard dataset variants (default: use filtered variants - MathVerse Text Lite only and DynaMath varid=1 only)")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of worker threads for parallel processing")
    parser.add_argument("--load_battles", action="store_true", 
                    help="Skip conducting battles and load existing battle results from output directory")
    args = parser.parse_args()
    
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all datasets using specialized loaders
    logging.info(f"Loading datasets from {args.data_path}")
    all_samples = {}
    
    # Load each requested dataset
    for dataset_name in args.datasets:
        try:
            if dataset_name == "wemath":
                all_samples["wemath"] = load_wemath_dataset(args.data_path, load_ratings=False)
                logging.info(f"Loaded {len(all_samples['wemath'])} samples from WeMath")
            
            elif dataset_name == "mathverse":
                # Use filtered_variants to control text_lite_only
                all_samples["mathverse"] = load_mathverse_dataset(
                    args.data_path, 
                    text_lite_only=args.use_filtered_variants,
                    load_ratings=False
                )
                variant_desc = " (Text Lite only)" if args.use_filtered_variants else " (all variants)"
                logging.info(f"Loaded {len(all_samples['mathverse'])} samples from MathVerse{variant_desc}")
                
            elif dataset_name == "dynamath":
                # Use filtered_variants to control filter_varid1
                all_samples["dynamath"] = load_dynamath_dataset(
                    args.tsv_data_path, 
                    args.data_path,
                    filter_varid1=args.use_filtered_variants,
                    load_ratings=False
                )
                variant_desc = " (varid=1 only)" if args.use_filtered_variants else " (all variants)"
                logging.info(f"Loaded {len(all_samples['dynamath'])} samples from DynaMath{variant_desc}")
                
            elif dataset_name == "mathvision":
                all_samples["mathvision"] = load_mathvision_dataset(args.tsv_data_path, args.data_path, load_ratings=False)
                logging.info(f"Loaded {len(all_samples['mathvision'])} samples from MathVision")
                
            elif dataset_name == "mathvista":
                all_samples["mathvista"] = load_mathvista_dataset(args.data_path, load_ratings=False)
                logging.info(f"Loaded {len(all_samples['mathvista'])} samples from MathVista")
                
            else:
                logging.warning(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            logging.error(f"Error loading {dataset_name} dataset: {e}")
    
    # Conduct battles within each dataset separately

    all_dataset_battles = {} # Initialize the dictionary

    if args.load_battles:
        logging.info("--- Loading existing battle results ---")
        for dataset_name in args.datasets:
            if dataset_name not in all_samples:
                logging.warning(f"Dataset '{dataset_name}' was specified but not loaded. Skipping battle loading.")
                continue
                
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            battles_file_path = os.path.join(dataset_output_dir, "battles.csv")
            
            if os.path.exists(battles_file_path):
                try:
                    logging.info(f"Loading battles for {dataset_name} from {battles_file_path}")
                    battles_df = pd.read_csv(
                        battles_file_path, 
                        dtype={'sample_a_id': str, 'sample_b_id': str} 
                    )
                    # Ensure required columns exist
                    if not all(col in battles_df.columns for col in ["sample_a_id", "sample_b_id", "winner"]):
                         logging.error(f"Battles file {battles_file_path} is missing required columns. Skipping.")
                         continue
                    all_dataset_battles[dataset_name] = battles_df
                except Exception as e:
                    logging.error(f"Error loading battles file {battles_file_path}: {e}. Skipping dataset.")
            else:
                logging.warning(f"Battles file not found for {dataset_name} at {battles_file_path}. Skipping dataset.")
                # Optionally, create an empty DataFrame if you want processing to continue
                # all_dataset_battles[dataset_name] = pd.DataFrame(columns=["sample_a_id", "sample_b_id", "winner"])

    else:
        logging.info("--- Conducting new battles ---")
        # Conduct battles within each dataset separately
        # NOTE: Using a fixed opponents_per_sample=32 here as in the original code. 
        # Consider making this an argument if needed.
        conducted_battles = conduct_battles(all_samples, opponents_per_sample=32, args=args) 
        
        # Save the newly conducted battles
        for dataset_name, battles_df in conducted_battles.items():
             # Create dataset-specific output directory
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Save raw battle results for this dataset
            battles_file_path = os.path.join(dataset_output_dir, "battles.csv")
            logging.info(f"Saving conducted battles for {dataset_name} to {battles_file_path}")
            battles_df.to_csv(battles_file_path, index=False)
            
            # Add to the dictionary for rating computation
            all_dataset_battles[dataset_name] = battles_df


    # --- Processing and saving results ---
    logging.info("--- Computing ratings and saving results ---")
    for dataset_name, battles_df in all_dataset_battles.items():
        # Check if DataFrame is valid and not empty before proceeding
        if battles_df is None or battles_df.empty:
             logging.warning(f"No valid battle data found for {dataset_name}. Skipping rating computation.")
             continue

        # Ensure the output directory exists
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        logging.info(f"Processing ratings for {dataset_name} ({len(battles_df)} battles)")

        # Define the default rating value (initial rating used in compute_bt_ratings)
        default_rating = 1000.0 # Use float for consistency

        try:
            # Compute Bradley-Terry ratings (point estimate first)
            ratings = compute_bt_ratings(battles_df)
            if ratings.empty:
                 logging.warning(f"Initial rating computation returned empty results for {dataset_name}. Skipping CI and ranking for this dataset.")
                 # Decide how to handle this - maybe assign default to all?
                 # For now, we'll skip, but you might want to assign default 1000 to all samples here.
                 continue

            # Compute bootstrap confidence intervals if enough data
            ratings_with_ci = pd.DataFrame() # Initialize empty DataFrame
            if len(battles_df) >= 10: # Use actual battles_df length
                logging.info(f"Computing bootstrap CIs for {dataset_name}...")
                ratings_with_ci = compute_bootstrap_confidence_intervals(battles_df, args.bootstrap_rounds, 8) # Use args for rounds/workers
            else:
                logging.warning(f"Not enough battles ({len(battles_df)}) for bootstrap CI in {dataset_name}. Using point estimates for CI.")
                # Create DataFrame with point estimates if no CI
                ratings_with_ci = pd.DataFrame({
                    'rating': ratings,
                    'lower_ci': ratings, # Fill CI with rating itself
                    'upper_ci': ratings
                })

            # --- Prepare a complete DataFrame for ALL samples in the dataset ---
            if dataset_name in all_samples:
                current_samples = all_samples[dataset_name]
                all_sample_ids = [s.get('id') for s in current_samples if s.get('id') is not None]

                if not all_sample_ids:
                    logging.warning(f"No valid sample IDs found in original data for {dataset_name}. Cannot proceed.")
                    continue

                # Create a DataFrame indexed by all valid sample IDs from the original dataset
                complete_results_df = pd.DataFrame(index=pd.Index(all_sample_ids, name='id'))

                # Join the calculated ratings and CIs (if any)
                # Keep all original IDs using a left join
                if not ratings_with_ci.empty:
                    complete_results_df = complete_results_df.join(ratings_with_ci[['rating', 'lower_ci', 'upper_ci']])
                else:
                    # If ratings_with_ci was empty (e.g., bootstrap failed), create columns to fill
                    complete_results_df['rating'] = np.nan
                    complete_results_df['lower_ci'] = np.nan
                    complete_results_df['upper_ci'] = np.nan


                # *** Fill missing ratings/CIs with the default value ***
                # Samples not in ratings_with_ci will have NaN here
                fill_values = {'rating': default_rating, 'lower_ci': default_rating, 'upper_ci': default_rating}
                complete_results_df.fillna(value=fill_values, inplace=True)
                logging.info(f"Filled {complete_results_df['rating'].isna().sum()} missing ratings with default value {default_rating} before ranking.")


                # *** Calculate ranks on the COMPLETE DataFrame ***
                # Now includes samples that were initially unrated but assigned the default
                complete_results_df['rank'] = complete_results_df['rating'].rank(
                    ascending=False, # Higher rating = better rank (rank 1)
                    method='min'     # Assign the minimum rank in case of ties (e.g., all 1000s get the same rank)
                ).astype(int)

                # --- Update the original sample data using the complete results ---
                updated_samples_count = 0
                for sample in current_samples:
                    sample_id = sample.get('id')
                    if sample_id is not None and sample_id in complete_results_df.index:
                        try:
                            # Look up results in the complete DataFrame
                            sample['rating'] = float(complete_results_df.loc[sample_id, 'rating'])
                            sample['rank'] = int(complete_results_df.loc[sample_id, 'rank']) # Rank is now guaranteed
                            sample['lower_ci'] = float(complete_results_df.loc[sample_id, 'lower_ci'])
                            sample['upper_ci'] = float(complete_results_df.loc[sample_id, 'upper_ci'])
                            updated_samples_count += 1
                        except (KeyError, ValueError) as e:
                             logging.warning(f"Error updating sample {sample_id} in {dataset_name} from complete results: {e}. Setting fields to None.")
                             # Assign None if lookup/conversion fails unexpectedly
                             sample['rating'] = None
                             sample['rank'] = None
                             sample['lower_ci'] = None
                             sample['upper_ci'] = None
                    else:
                        # Handle samples originally without an ID
                        logging.warning(f"Sample without ID found in {dataset_name}. Cannot assign rating/rank.")
                        sample['rating'] = None
                        sample['rank'] = None
                        sample['lower_ci'] = None
                        sample['upper_ci'] = None

                logging.info(f"Updated ratings and ranks for {updated_samples_count}/{len(current_samples)} samples in {dataset_name}.")

                # Save updated samples with ratings for this dataset
                samples_output_path = os.path.join(dataset_output_dir, "samples_with_ratings.json")
                logging.info(f"Saving updated samples for {dataset_name} to {samples_output_path}")
                with open(samples_output_path, 'w') as f:
                    json.dump(current_samples, f, indent=2)
            else:
                 logging.warning(f"Original samples for {dataset_name} not found in 'all_samples'. Cannot save updated samples.")

        except Exception as e:
             logging.error(f"An error occurred during rating computation or processing for {dataset_name}: {e}", exc_info=True) # Add traceback


    logging.info("All processing finished.")

    return 0


if __name__ == "__main__":
    sys.exit(main())