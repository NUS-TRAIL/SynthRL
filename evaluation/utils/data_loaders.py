import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset

def load_geo3k_dataset(data_path: str) -> List[Dict]:
    """Load Geo3K dataset"""
    data_path = os.path.join(data_path, "geometry3k/test")
    dataset = []
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for folder in tqdm(folders, desc="Loading Geo3K data"):
        folder_path = os.path.join(data_path, folder)
        image_path = os.path.join(folder_path, "img_diagram.png")
        json_path = os.path.join(folder_path, "data.json")
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            continue
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        dataset.append({
            "id": data["id"],
            "image_path": image_path,
            "question": data["annotat_text"],
            "answer": data["choices"][mapping[data["answer"]]],
            "dataset": "geo3k"
        })
    
    return dataset

def load_logicvista_dataset(data_path: str) -> List[Dict]:
    """Load LogicVista dataset"""
    image_root_path = os.path.join(data_path, "logicvista/images")
    data_path = os.path.join(data_path, "logicvista/dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for idx, item in data.items():
        image_path = os.path.join(image_root_path, item["imagename"])
        dataset.append({
            "id": idx,
            "image_path": image_path,
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "logicvista"
        })
    
    return dataset

def load_wemath_dataset(data_path: str, load_ratings: bool = True, low: float = 950, high: float = 1050) -> List[Dict]:
    """
    Load WeMath dataset
    
    Args:
        data_path: Base directory containing dataset files
        load_ratings: If True, load and merge ratings data
        low: Threshold for 'easy' level (rating <= low)
        high: Threshold for 'hard' level (rating >= high)
    
    Returns:
        List of problem dictionaries
    """
    image_root = os.path.join(data_path, "wemath/images")
    json_path = os.path.join(data_path, "wemath/testmini.json")
    rankings_path = os.path.join(data_path, "wemath/samples_with_ratings.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load rankings data if enabled
    rankings_map = {}
    if load_ratings:
        with open(rankings_path, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        # Create a mapping of id -> rankings
        rankings_map = {item.get("id", ""): {
                            "rating": item.get("rating"),
                            "rank": item.get("rank")
                        } 
                        for item in rankings if "id" in item}
    
    dataset = []
    for item in data:
        # Determine the image path
        image_path = os.path.join(image_root, item["image_path"])
        
        # Create the item ID
        item_id = item["ID"] + "@" + item["key"]
        
        # Create the dataset item
        dataset_item = {
            "id": item_id,
            "image_path": image_path,
            "question": f"{item['question']}\n\nOptions: {item['option']}",
            "answer": item["answer"],
            "dataset": "wemath"
        }
        
        # Add rating and rank if available and enabled
        if load_ratings and item_id in rankings_map:
            dataset_item["rating"] = rankings_map[item_id]["rating"]
            dataset_item["rank"] = rankings_map[item_id]["rank"]
            
            # Add level based on rating
            rating = rankings_map[item_id]["rating"]
            if rating >= high:
                dataset_item["level"] = "hard"
            elif rating <= low:
                dataset_item["level"] = "easy"
            else:
                dataset_item["level"] = "medium"
        
        dataset.append(dataset_item)
    
    return dataset

def load_mathvista_dataset(data_path: str, load_ratings: bool = True, low: float = 950, high: float = 1050) -> List[Dict]:
    """
    Load MathVista dataset
    
    Args:
        data_path: Base directory containing dataset files
        load_ratings: If True, load and merge ratings data
        low: Threshold for 'easy' level (rating <= low)
        high: Threshold for 'hard' level (rating >= high)
        
    Returns:
        List of problem dictionaries
    """
    image_base_dir = os.path.join(data_path, "mathvista")
    dataset_raw = load_dataset("AI4Math/MathVista", split="testmini")
    rankings_path = os.path.join(data_path, "mathvista/samples_with_ratings.json")
    
    # Load rankings data if enabled
    rankings_map = {}
    if load_ratings:
        with open(rankings_path, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        # Create a mapping of id -> ratings data
        rankings_map = {}
        for item in rankings:
            if "id" in item:
                item_id = item["id"]
                rankings_map[item_id] = {
                    "rating": item.get("rating"),
                    "rank": item.get("rank")
                }
    
    dataset = []
    mapping = {
        "0": "A", "1": "B", "2": "C", "3": "D",
        "4": "E", "5": "F", "6": "G", "7": "H"
    }
    
    for item in dataset_raw:
        if item["question_type"] == "multi_choice":
            idx = item["choices"].index(item["answer"])
            answer = mapping[str(idx)]
        else:
            answer = item["answer"]
        
        item_id = item.get("pid", "")
        
        # Create the dataset item
        dataset_item = {
            "id": item_id,
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query"],
            "answer": answer,
            "task": item["metadata"]["task"],
            "dataset": "mathvista"
        }
        
        # Add rating and rank if available and enabled
        if load_ratings and item_id in rankings_map:
            dataset_item["rating"] = rankings_map[item_id]["rating"]
            dataset_item["rank"] = rankings_map[item_id]["rank"]
            
            # Add level based on rating
            rating = rankings_map[item_id]["rating"]
            if rating >= high:
                dataset_item["level"] = "hard"
            elif rating <= low:
                dataset_item["level"] = "easy"
            else:
                dataset_item["level"] = "medium"
        
        dataset.append(dataset_item)
    
    return dataset

def load_mathverse_dataset(data_path: str, text_lite_only: bool = False, load_ratings: bool = True, 
                          low: float = 950, high: float = 1050) -> List[Dict]:
    """
    Load MathVerse dataset
    
    Args:
        data_path: Base directory containing dataset files
        text_lite_only: If True, only include problems with problem_version="Text Lite"
                        and use problem_index as the id for mapping
        load_ratings: If True, load and merge ratings data
        low: Threshold for 'easy' level (rating <= low)
        high: Threshold for 'hard' level (rating >= high)
        
    Returns:
        List of problem dictionaries
    """
    image_base_dir = os.path.join(data_path, "mathverse/images")
    json_path = os.path.join(data_path, "mathverse/testmini.json")
    rankings_path = os.path.join(data_path, "mathverse/samples_with_ratings.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load rankings data if enabled
    rankings_map = {}
    if load_ratings:
        with open(rankings_path, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        # Create a mapping of id -> ratings data
        rankings_map = {}
        for item in rankings:
            if "id" in item:
                # For mathverse, use problem_index for mapping
                item_id = str(item.get("id", ""))
                rankings_map[item_id] = {
                    "rating": item.get("rating"),
                    "rank": item.get("rank")
                }
    
    dataset = []
    for item in data:
        # Skip if we only want Text Lite and this isn't one
        if text_lite_only and item.get("problem_version") != "Text Lite":
            continue
        
        # Determine which ID to use for displaying/referencing
        if text_lite_only:
            display_id = str(item.get("problem_index", ""))
        else:
            display_id = str(item.get("sample_index", ""))
            
        # For mathverse, always use problem_index to look up ratings
        rating_lookup_id = str(item.get("problem_index", ""))
        
        # Create the dataset item
        dataset_item = {
            "id": display_id,  # ID for display/reference
            'sample_index': item.get("sample_index", ""),  # Keep original sample_index for reference
            'problem_index': item.get("problem_index", ""),  # Keep original problem_index for reference
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query_cot"],
            "question_for_eval": item["question_for_eval"],
            "answer": item["answer"],
            "problem_version": item["problem_version"],
            "dataset": "mathverse"
        }
        
        # Add rating and rank if available and enabled
        if load_ratings and rating_lookup_id in rankings_map:
            dataset_item["rating"] = rankings_map[rating_lookup_id]["rating"]
            dataset_item["rank"] = rankings_map[rating_lookup_id]["rank"]
            
            # Add level based on rating
            rating = rankings_map[rating_lookup_id]["rating"]
            if rating >= high:
                dataset_item["level"] = "hard"
            elif rating <= low:
                dataset_item["level"] = "easy"
            else:
                dataset_item["level"] = "medium"
        
        dataset.append(dataset_item)
    
    return dataset

def load_mathvision_dataset(tsv_path: str, data_path: str, load_ratings: bool = True, 
                           low: float = 950, high: float = 1050) -> List[Dict]:
    """
    Load MathVision dataset
    
    Args:
        tsv_path: Path to the directory containing the TSV file
        data_path: Path to the directory containing images
        load_ratings: If True, load and merge ratings data
        low: Threshold for 'easy' level (rating <= low)
        high: Threshold for 'hard' level (rating >= high)
        
    Returns:
        List of problem dictionaries
    """
    tsv_file_path = os.path.join(tsv_path, "MathVision.tsv")
    image_base_dir = os.path.join(data_path, "mathvision/images")
    rankings_path = os.path.join(data_path, "mathvision/samples_with_ratings.json")
    
    # Load dataset
    df = pd.read_csv(tsv_file_path, sep='\t')
    
    # Load rankings data if enabled
    rankings_map = {}
    if load_ratings:
        with open(rankings_path, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        # Create a mapping of id -> ratings data
        rankings_map = {}
        for item in rankings:
            if "id" in item:
                item_id = str(item["id"])
                rankings_map[item_id] = {
                    "rating": item.get("rating"),
                    "rank": item.get("rank")
                }
    
    dataset = []
    for _, row in df.iterrows():
        item_id = str(row.get("index", ""))
        
        # Create the dataset item
        dataset_item = {
            "id": item_id,
            "image_path": os.path.join(image_base_dir, f"{row['index']}.jpg"),
            "question": row["question"],
            "answer": row["answer"],
            "subject": row.get("category", "unknown"),
            "dataset": "mathvision"
        }
        
        # Add rating and rank if available and enabled
        if load_ratings and item_id in rankings_map:
            dataset_item["rating"] = rankings_map[item_id]["rating"]
            dataset_item["rank"] = rankings_map[item_id]["rank"]
            
            # Add level based on rating
            rating = rankings_map[item_id]["rating"]
            if rating >= high:
                dataset_item["level"] = "hard"
            elif rating <= low:
                dataset_item["level"] = "easy"
            else:
                dataset_item["level"] = "medium"
        
        dataset.append(dataset_item)
    
    return dataset

def load_dynamath_dataset(tsv_path: str, data_path: str, filter_varid1: bool = False, 
                         load_ratings: bool = True, low: float = 950, high: float = 1050) -> List[Dict]:
    """
    Load Dynamath dataset
    
    Args:
        tsv_path: Path to the directory containing the TSV file
        data_path: Path to the directory containing images
        filter_varid1: If True, only include rows where varid=1
                      and use qid as the id for mapping
        load_ratings: If True, load and merge ratings data
        low: Threshold for 'easy' level (rating <= low)
        high: Threshold for 'hard' level (rating >= high)
        
    Returns:
        List of problem dictionaries
    """
    tsv_file_path = os.path.join(tsv_path, "DynaMath.tsv")
    image_base_dir = os.path.join(data_path, "dynamath/images")
    rankings_path = os.path.join(data_path, "dynamath/samples_with_ratings.json")
    
    # Load dataset
    df = pd.read_csv(tsv_file_path, sep='\t')
    
    # Filter by varid=1 if requested
    if filter_varid1:
        df = df[df['varid'] == 1]
    
    # Load rankings data if enabled
    rankings_map = {}
    if load_ratings:
        with open(rankings_path, "r", encoding="utf-8") as f:
            rankings = json.load(f)
        
        # Create a mapping of id -> ratings data
        rankings_map = {}
        for item in rankings:
            if "id" in item:
                item_id = str(item["id"])
                rankings_map[item_id] = {
                    "rating": item.get("rating"),
                    "rank": item.get("rank")
                }
    
    dataset = []
    for _, row in df.iterrows():
        question = row["question"]
        
        # Check if answer_type is float
        if row.get("answer_type") == 'float':
            # Check if the instruction is already in the question
            if "Answer the question with a floating-point number." in question:
                # Replace with the three-digit version
                question = question.replace(
                    "Answer the question with a floating-point number.",
                    "Answer the question with a three-digit floating-point number."
                )
            else:
                # Add the instruction if it's not there
                question = question + " Answer the question with a three-digit floating-point number."
        
        # Determine which ID to use for display/reference
        if filter_varid1:
            display_id = str(row.get("qid", ""))
        else:
            display_id = str(row.get("index", ""))
        
        # For dynamath, always use qid to look up ratings
        rating_lookup_id = str(row.get("qid", ""))
        
        # Create the dataset item
        dataset_item = {
            "id": display_id,  # ID for display/reference
            "index": str(row.get("index", "")),  # Keep original index for reference
            "image_path": os.path.join(image_base_dir, f"{row['index']}.jpg"),
            "question": question,
            "answer": row["answer"],
            "answer_type": row.get("answer_type", "text"),
            "dataset": "dynamath"
        }
        
        # Add rating and rank if available and enabled
        if load_ratings and rating_lookup_id in rankings_map:
            dataset_item["rating"] = rankings_map[rating_lookup_id]["rating"]
            dataset_item["rank"] = rankings_map[rating_lookup_id]["rank"]
            
            # Add level based on rating
            rating = rankings_map[rating_lookup_id]["rating"]
            if rating >= high:
                dataset_item["level"] = "hard"
            elif rating <= low:
                dataset_item["level"] = "easy"
            else:
                dataset_item["level"] = "medium"
        
        dataset.append(dataset_item)
    
    return dataset

def load_olympiadbench_dataset(data_dir: str) -> List[Dict]:
    """Load OlympiadBench dataset from metadata JSON with explicitly specified fields"""
    metadata_path = os.path.join(data_dir, "olympiadbench/olympiadbench_metadata.json")
    image_dir = os.path.join(data_dir, "olympiadbench/images")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    dataset = []
    for item in metadata:
        image_filenames = item.get("image_paths", [])
        
        # Handle image_path based on the type of data
        if image_filenames == ["invalid"]:
            # Text-only data
            image_path = ["text-only"]
        elif isinstance(image_filenames, list) and image_filenames:
            # Multimodal with images - create list of full paths
            image_path = [os.path.join(image_dir, filename) for filename in image_filenames]
        else:
            # Empty list - no images were processed
            image_path = None
        
        dataset.append({
            "id": item.get("id", ""),
            "image_path": image_path,
            "question": item.get("question", ""),
            "context": item.get("context", ""),
            "answer": item.get("answer", ""),
            "answer_type": item.get("answer_type", ""),
            "source": item.get("source", ""),
            "subject": item.get("subject", ""),
            "image_paths": item.get("image_paths", []),
            "dataset": "OlympiadBench"
        })
    
    return dataset

def load_hallubench_dataset(data_path: str) -> List[Dict]:
    """Load Hallubench dataset"""
    image_base_dir = os.path.join(data_path, "hallubench/images")
    data_path = os.path.join(data_path, "hallubench/HallusionBench.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        if not item["filename"]:
            continue
        
        if "?" in item["question"]:
            question = item["question"].split("?")[:-1][0]
        else:
            question = item["question"]
        question += "? You final answer can only be \\boxed{yes} or \\boxed{no}."
        gt_answer = "yes" if int(item["gt_answer"]) == 1 else "no"
        sid, fid, qid = item["set_id"], item["figure_id"], item["question_id"]
        dataset.append({
            "id": f"{sid}_{fid}_{qid}",
            "image_path": os.path.join(image_base_dir, item["filename"].replace("./", "")),
            "question": question,
            "question_for_eval": question,
            "answer": gt_answer,
            "problem_version": item["subcategory"],
            "dataset": "hallubench"
        })
    
    return dataset