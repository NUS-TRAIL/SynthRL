from datasets import load_from_disk, Dataset, Features, Image as HFImage, Value, Sequence
import argparse
import os
from tqdm import tqdm
from PIL import Image
import io

def convert_dataset(input_path, info, use_original_as_fallback=False, args=None, lower_threshold=None, upper_threshold=None):
    # Load dataset from disk
    print(f"Loading dataset from ./synthrl_data/{input_path}")
    dataset = load_from_disk(f'./synthrl_data/{input_path}')

    # Filter out samples with progress bar
    print(f"Filtering dataset with {len(dataset)} samples...")
    has_evolved = []
    has_actual_evolved = []  # Track samples with actual evolved questions (no fallback)
    no_evolved_but_kept = []  # Track samples with no evolved question but kept due to fallback

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Filtering"):
        # Check if evolved_question exists and is not None
        if 'evolved_question' in sample and sample['evolved_question'] is not None:
            has_evolved.append(i)
            has_actual_evolved.append(i)
        elif use_original_as_fallback:
            has_evolved.append(i)
            no_evolved_but_kept.append(i)

    valid_samples = dataset.select(has_evolved)
    actual_evolved_samples = dataset.select(has_actual_evolved)

    print(f"Filtered to {len(valid_samples)} valid samples")
    print(f"Samples with actual evolved questions: {len(actual_evolved_samples)}")

    if use_original_as_fallback:
        print(f"Including {len(no_evolved_but_kept)} samples using original question as fallback")

    # Create original and evolved datasets with a single mapping operation
    def transform_data(example):
        original_problem = example['original_question']

        # Use the original as evolved if no evolved question and fallback enabled
        if 'evolved_question' in example and example['evolved_question'] is not None:
            evolved_problem = example['evolved_question']
            is_fallback = False
        else:
            evolved_problem = original_problem
            is_fallback = True

        # Convert image to PIL.Image if it's not already
        pil_image = example['image']
        if not isinstance(pil_image, Image.Image):
            # If it's bytes or something else, try to convert to PIL Image
            if isinstance(pil_image, bytes):
                pil_image = Image.open(io.BytesIO(pil_image))

        # Process original problem
        original_valid = True
        if original_problem.count("<image>") == 0:
            # Add image tag at the beginning if missing
            original_problem = "<image>\n" + original_problem
        elif original_problem.count("<image>") > 1:
            # Discard if multiple image tags
            original_valid = False

        # Process evolved problem
        evolved_valid = True
        if evolved_problem.count("<image>") == 0:
            # Add image tag at the beginning if missing
            evolved_problem = "<image>\n" + evolved_problem
        elif evolved_problem.count("<image>") > 1:
            # Discard if multiple image tags
            evolved_valid = False

        # Get pass values if they exist
        original_pass = example.get('original_pass', 0)
        evolved_pass = example.get('evolved_pass', 0)

        return {
            'original': {
                'id': example['id'],
                'images': [pil_image],  # Using PIL Image object and renamed to "images"
                'problem': original_problem,
                'answer': example['original_answer'],
                'valid': original_valid,
                'pass_rates': original_pass,
                'is_evolved': False
            },
            'evolved': {
                'id': example['id'],
                'images': [pil_image],  # Using PIL Image object and renamed to "images"
                'problem': evolved_problem,
                'answer': example['original_answer'],
                'valid': evolved_valid,
                'is_fallback': is_fallback,
                'pass_rates': evolved_pass,
                'is_evolved': True
            },
            'has_actual_evolved': not is_fallback,
        }

    # Transform in a single pass
    transformed = valid_samples.map(
        transform_data,
        num_proc=64,  # Specify number directly to avoid overhead
        desc="Transforming dataset"
    )

    # Extract and filter the original and evolved datasets
    def extract_original(example):
        result = {
            'id': example['original']['id'],
            'images': example['original']['images'],
            'problem': example['original']['problem'],
            'answer': example['original']['answer'],
            'pass_rates': example['original']['pass_rates'],
            'is_evolved': False
        }
        return result, example['original']['valid'] and example['evolved']['valid']

    def extract_evolved(example):
        result = {
            'id': example['evolved']['id'],
            'images': example['evolved']['images'],
            'problem': example['evolved']['problem'],
            'answer': example['evolved']['answer'],
            'pass_rates': example['evolved']['pass_rates'],
            'is_evolved': True,
            'is_fallback': example['evolved']['is_fallback']
        }
        return result, example['original']['valid'] and example['evolved']['valid']

    # Apply extraction and filtering for original dataset
    print("Extracting and filtering original dataset...")
    original_data = []
    for item in tqdm(transformed, desc="Processing original"):
        entry, is_valid = extract_original(item)
        if is_valid:
            original_data.append(entry)

    # Apply extraction and filtering for evolved dataset
    print("Extracting and filtering evolved dataset...")
    evolved_data = []
    actual_evolved_data = []  # Store only non-fallback evolved samples

    for item in tqdm(transformed, desc="Processing evolved"):
        entry, is_valid = extract_evolved(item)
        if is_valid:
            evolved_data.append(entry)
            if not entry.get('is_fallback', False):
                # Store only non-fallback evolved samples separately
                evolved_entry = entry.copy()
                if 'is_fallback' in evolved_entry:
                    del evolved_entry['is_fallback']
                actual_evolved_data.append(evolved_entry)

    # Create combined dataset: all originals + all actual evolved samples
    combined_data = []
    # First add all original samples
    combined_data.extend(original_data)
    # Then add all actual evolved samples (without fallbacks)
    combined_data.extend(actual_evolved_data)

    # Clean up 'is_fallback' from evolved_data for consistency
    for entry in evolved_data:
        if 'is_fallback' in entry:
            del entry['is_fallback']

    # Define features to ensure proper image handling
    features = Features({
        'id': Value('string'),
        'images': Sequence(HFImage()),
        'problem': Value('string'),
        'answer': Value('string'),
        'pass_rates': Value('int32'),
        'is_evolved': Value('bool')
    })

    # Create datasets from filtered data with explicit features
    original_dataset = Dataset.from_list(original_data, features=features)
    combined_dataset = Dataset.from_list(combined_data, features=features)
    
    print(f"After image tag filtering: {len(original_dataset)} valid original samples")
    print(f"Combined dataset (all original + actual evolved): {len(combined_dataset)} samples")

    # Helper function for creating train/test splits
    def create_train_test_split(dataset, test_size=0, random_seed=42):
        # Shuffle the dataset for random sampling
        shuffled_indices = list(range(len(dataset)))
        
        import random
        random.seed(random_seed)
        random.shuffle(shuffled_indices)
        
        # Select test and train indices
        test_indices = shuffled_indices[:min(test_size, len(dataset))]
        train_indices = shuffled_indices[min(test_size, len(dataset)):]
        
        # Create the splits
        train_dataset = dataset.select(train_indices)
        test_dataset = dataset.select(test_indices)
        
        return train_dataset, test_dataset
    
    # Create train/test splits
    original_train, original_test = create_train_test_split(original_dataset)
    combined_train, combined_test = create_train_test_split(combined_dataset)
    
    # Create output directories
    output_dirs = [
        f'./data/verifiable_data/{info}_original/train',
        f'./data/verifiable_data/{info}_original/test',
        f'./data/verifiable_data/{info}_combined/train',
        f'./data/verifiable_data/{info}_combined/test',
    ]

    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # Save datasets in parquet format
    print("Saving datasets to parquet files...")
    original_train.to_parquet(f'./data/verifiable_data/{info}_original/train/data.parquet')
    original_test.to_parquet(f'./data/verifiable_data/{info}_original/test/data.parquet')
    combined_train.to_parquet(f'./data/verifiable_data/{info}_combined/train/data.parquet')
    combined_test.to_parquet(f'./data/verifiable_data/{info}_combined/test/data.parquet')

    print(f"Original dataset: {len(original_train)} train, {len(original_test)} test samples")
    print(f"Combined dataset: {len(combined_train)} train, {len(combined_test)} test samples")

    return original_dataset, combined_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Huggingface dataset format')
    parser.add_argument('--input_path', type=str, required=True, help='Input path relative to ./synthrl_data/')
    parser.add_argument('--info', type=str, required=True, help='Info tag for output directory naming')
    parser.add_argument('--use_original_as_fallback', action='store_true',
                        help='If true, use original question as evolved when no evolved questions exist')
    args = parser.parse_args()

    original_dataset, combined_dataset = convert_dataset(
        args.input_path,
        args.info,
        args.use_original_as_fallback,
        args
    )