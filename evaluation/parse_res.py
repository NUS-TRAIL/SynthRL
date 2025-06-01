#!/usr/bin/env python3
import os
import json
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_step_number(step_dir):
    """Extract the step number from directory name."""
    match = re.search(r'global_step_(\d+)', step_dir)
    if match:
        return int(match.group(1))
    return 0

def process_benchmark_file(json_file):
    """Extract metrics from a benchmark JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract the main accuracy
        accuracy = data.get('metrics', {}).get('accuracy', None)
        
        # Extract sub-accuracies if available
        sub_accuracies = data.get('metrics', {}).get('sub_accuracies', {})
        
        return {
            'accuracy': accuracy,
            'sub_accuracies': sub_accuracies
        }
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {'accuracy': None, 'sub_accuracies': {}}

def parse_model_benchmarks(base_path, model_dir):
    """Parse all benchmarks for a specific model."""
    model_path = os.path.join(base_path, model_dir)
    
    # Find all global_step directories
    step_dirs = sorted(glob.glob(os.path.join(model_path, 'global_step_*')), 
                      key=extract_step_number)
    
    if not step_dirs:
        print(f"No global_step directories found in {model_path}")
        return None
    
    # Dictionary to track metrics for each benchmark and step
    # Using a more flexible structure to handle missing data
    benchmark_data = {
        'main': {},  # Main accuracy
        'hard': {},  # Hard difficulty
        'medium': {},  # Medium difficulty
        'easy': {}  # Easy difficulty
    }
    
    steps = []  # Track all steps
    
    # Process each global step directory
    for step_dir in step_dirs:
        step_num = extract_step_number(step_dir)
        steps.append(step_num)
        
        # Find all JSON files (benchmark results)
        json_files = glob.glob(os.path.join(step_dir, '*.json'))
        
        for json_file in json_files:
            benchmark_name = os.path.basename(json_file).replace('.json', '')
            
            # Skip geo3k as requested
            if benchmark_name in ['geo3k']:
                continue
            
            metrics = process_benchmark_file(json_file)
            
            # Initialize benchmark data if needed
            for data_type in ['main', 'hard', 'medium', 'easy']:
                if benchmark_name not in benchmark_data[data_type]:
                    benchmark_data[data_type][benchmark_name] = {}
            
            # Store main accuracy
            if metrics['accuracy'] is not None:
                benchmark_data['main'][benchmark_name][step_num] = metrics['accuracy']
            
            # Store sub-accuracies
            for sub_key, sub_value in metrics['sub_accuracies'].items():
                if sub_key == 'hard' and sub_value is not None:
                    benchmark_data['hard'][benchmark_name][step_num] = sub_value
                elif sub_key == 'medium' and sub_value is not None:
                    benchmark_data['medium'][benchmark_name][step_num] = sub_value
                elif sub_key == 'easy' and sub_value is not None:
                    benchmark_data['easy'][benchmark_name][step_num] = sub_value
    
    # Convert to DataFrames with steps as index first
    df_main = pd.DataFrame({'step': steps})
    df_hard = pd.DataFrame({'step': steps})
    df_medium = pd.DataFrame({'step': steps})
    df_easy = pd.DataFrame({'step': steps})
    
    # Add benchmark data to dataframes
    for benchmark in benchmark_data['main']:
        if benchmark_data['main'][benchmark]:  # If we have data for this benchmark
            # Convert the dictionary to a series with appropriate index
            benchmark_series = pd.Series(benchmark_data['main'][benchmark])
            # Create a column in the DataFrame, filling missing values with NaN
            df_main[benchmark] = df_main['step'].map(benchmark_data['main'][benchmark]).values
    
    for benchmark in benchmark_data['hard']:
        if benchmark_data['hard'][benchmark]:
            df_hard[benchmark] = df_hard['step'].map(benchmark_data['hard'][benchmark]).values
    
    for benchmark in benchmark_data['medium']:
        if benchmark_data['medium'][benchmark]:
            df_medium[benchmark] = df_medium['step'].map(benchmark_data['medium'][benchmark]).values
    
    for benchmark in benchmark_data['easy']:
        if benchmark_data['easy'][benchmark]:
            df_easy[benchmark] = df_easy['step'].map(benchmark_data['easy'][benchmark]).values
    
    # Calculate average of all benchmarks (excluding geo3k and 'step')
    benchmark_cols = [col for col in df_main.columns if col != 'step']
    if benchmark_cols:  # Only calculate if we have benchmark columns
        df_main['average'] = df_main[benchmark_cols].mean(axis=1)
    
    benchmark_cols = [col for col in df_hard.columns if col != 'step']
    if benchmark_cols:  # Only calculate if we have benchmark columns
        df_hard['average'] = df_hard[benchmark_cols].mean(axis=1)
    
    benchmark_cols = [col for col in df_medium.columns if col != 'step']
    if benchmark_cols:  # Only calculate if we have benchmark columns
        df_medium['average'] = df_medium[benchmark_cols].mean(axis=1)
    
    benchmark_cols = [col for col in df_easy.columns if col != 'step']
    if benchmark_cols:  # Only calculate if we have benchmark columns
        df_easy['average'] = df_easy[benchmark_cols].mean(axis=1)
    
    return df_main, df_hard, df_medium, df_easy

def save_model_csv(df_main, df_hard, df_medium, df_easy, base_path, model_name):
    """Save the DataFrames to CSV files for a specific model."""
    output_dir = os.path.join(base_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    main_file = os.path.join(output_dir, f'{model_name}_results.csv')
    df_main.to_csv(main_file, index=False)
    
    # Save difficulty-specific results
    hard_file = os.path.join(output_dir, f'{model_name}_results_hard.csv')
    df_hard.to_csv(hard_file, index=False)
    
    medium_file = os.path.join(output_dir, f'{model_name}_results_medium.csv')
    df_medium.to_csv(medium_file, index=False)
    
    easy_file = os.path.join(output_dir, f'{model_name}_results_easy.csv')
    df_easy.to_csv(easy_file, index=False)
    
    print(f"Results for {model_name} saved to:")
    print(f"  {main_file}")
    print(f"  {hard_file}")
    print(f"  {medium_file}")
    print(f"  {easy_file}")
    
    return main_file

def collect_all_models_data(base_path):
    """Collect data from all models for benchmarks comparison."""
    # Dictionary to store data for each benchmark across models
    all_benchmarks_data = {
        'main': {},  # Main benchmark results
        'hard': {},  # Hard difficulty results
        'medium': {}, # Medium difficulty results
        'easy': {}   # Easy difficulty results
    }
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.') and d != 'results']
    
    all_benchmarks = set()
    model_dataframes = {}
    
    # Process each model directory
    for model_dir in model_dirs:
        print(f"Processing model: {model_dir}")
        results = parse_model_benchmarks(base_path, model_dir)
        
        if results is not None:
            df_main, df_hard, df_medium, df_easy = results
            
            # Verify we have valid data (not empty dataframes)
            if not df_main.empty and 'step' in df_main.columns:
                model_dataframes[model_dir] = {
                    'main': df_main,
                    'hard': df_hard,
                    'medium': df_medium,
                    'easy': df_easy
                }
                
                # Save CSVs for this model
                save_model_csv(df_main, df_hard, df_medium, df_easy, base_path, model_dir)
                
                # Collect all benchmark names (excluding 'step' and 'average')
                for col in df_main.columns:
                    if col != 'step' and col != 'average':
                        all_benchmarks.add(col)
            else:
                print(f"Warning: No valid data found for model {model_dir}")
    
    return model_dataframes, all_benchmarks

def create_comparative_visualizations(model_dataframes, all_benchmarks, base_path):
    """Create comparative visualizations across all models."""
    output_dir = os.path.join(base_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Plot average performance across all models
    plot_comparative_metric('average', model_dataframes, 'main', output_dir, 'Average Performance Across Models')
    plot_comparative_metric('average', model_dataframes, 'hard', output_dir, 'Average Performance (Hard Difficulty)')
    plot_comparative_metric('average', model_dataframes, 'medium', output_dir, 'Average Performance (Medium Difficulty)')
    plot_comparative_metric('average', model_dataframes, 'easy', output_dir, 'Average Performance (Easy Difficulty)')
    
    # Plot each benchmark separately across all models
    for benchmark in all_benchmarks:
        plot_comparative_metric(benchmark, model_dataframes, 'main', output_dir, f'{benchmark.capitalize()} Performance Across Models')
        plot_comparative_metric(benchmark, model_dataframes, 'hard', output_dir, f'{benchmark.capitalize()} Performance (Hard Difficulty)')
        plot_comparative_metric(benchmark, model_dataframes, 'medium', output_dir, f'{benchmark.capitalize()} Performance (Medium Difficulty)')
        plot_comparative_metric(benchmark, model_dataframes, 'easy', output_dir, f'{benchmark.capitalize()} Performance (Easy Difficulty)')
        
    print(f"Comparative visualizations saved to {output_dir}")

def plot_comparative_metric(metric, model_dataframes, difficulty, output_dir, title):
    """Plot a specific metric across all models."""
    plt.figure(figsize=(12, 8))
    
    # Check if this metric exists in the dataframes
    has_data = False
    
    for model_name, dfs in model_dataframes.items():
        df = dfs[difficulty]
        if metric in df.columns and 'step' in df.columns:
            has_data = True
            
            # Extract data for plotting
            plot_data = df[['step', metric]].dropna()
            
            if not plot_data.empty:
                # Create a label using model name (shortened if too long)
                label = model_name
                if len(label) > 30:
                    label = label[:27] + "..."
                
                plt.plot(plot_data['step'], plot_data[metric], marker='o', linestyle='-', label=label)
    
    if not has_data:
        plt.close()
        return
    
    plt.title(title)
    plt.xlabel('Global Step')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Create a filename
    # Remove spaces and special characters from the title for the filename
    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    plt.savefig(os.path.join(output_dir, f'{filename}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Parse benchmark results and create summary CSV and visualizations.')
    parser.add_argument('--data-path', required=True, help='Path to the top-level directory (e.g., QWEN2.5-VERIFY-K12-V8)')
    args = parser.parse_args()
    
    # Full path to the specified directory
    base_path = os.path.join('logs_vlm', args.data_path)
    
    if not os.path.isdir(base_path):
        print(f"Directory not found: {base_path}")
        return
        
    # Process all models and collect data
    model_dataframes, all_benchmarks = collect_all_models_data(base_path)
    
    if model_dataframes:
        # Create comparative visualizations
        create_comparative_visualizations(model_dataframes, all_benchmarks, base_path)
    else:
        print("No data found to process")

if __name__ == "__main__":
    main()