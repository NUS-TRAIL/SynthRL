import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils.model_parser import llm_eval_score
from utils.math_grader import boxed_reward_fn
from mathruler.grader import extract_boxed_content, grade_answer

def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def format_text_only_prompt_olympiad(item: Dict, system_prompt: str):
    """Helper function to create text-only prompts with OlympiadBench-style formatting"""
    # Build the Olympiad-style question content
    olympiad_content = build_olympiad_content(item)
    
    # Format with your chat template
    prompt_text = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{system_prompt} {olympiad_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt_text


def format_prompt_with_images_olympiad(item: Dict, images: List, system_prompt: str):
    """Helper function to create properly formatted prompts with images using OlympiadBench-style formatting"""
    # Build the Olympiad-style question content
    olympiad_content = build_olympiad_content(item)
    
    # For a single image
    if len(images) == 1:
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{system_prompt} {olympiad_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        image_data = {"image": images[0]}
    # For multiple images
    else:
        vision_tags = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in range(len(images)))
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{vision_tags}{system_prompt} {olympiad_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        image_data = {"image": images}
    
    return prompt_text, image_data, len(images)


def build_olympiad_content(item: Dict):
    """Create Olympiad-style formatted content for the prompt"""
    # Determine subject and answer formatting
    is_math = "maths" in item.get("source", "").lower()
    subject_content = "Math" if is_math else "Physics"
    
    # Handle answer formatting
    is_multiple_answer = item.get("is_multiple_answer", False)
    if is_multiple_answer:
        answer_format = "\\boxed{multiple answers connected with commas}"
    else:
        answer_format = "\\boxed{answer}"
        
    # Handle units
    unit_text = ""
    if item.get("unit", ""):
        answer_format += "(unit)"
        unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
    
    # Get answer type description
    answer_type_text = get_answer_type_text(item.get("answer_type", ""), is_multiple_answer)
    
    # Build the OlympiadBench-style prompt
    olympiad_prompt = (
        f"The following is an open-ended problem from an International {subject_content} competition. "
        f"{answer_type_text}Please calculate the answer according to the given requirements and "
        "the information provided. Please use LaTeX format to represent the variables and formulas "
        'used in the solution process and results. Please end your solution with "So the final answer '
        f'is {answer_format}." and give the result explicitly{unit_text}.'
    )
    
    # Combine with context if available (for physics problems)
    if not is_math and "context" in item and str(item["context"]) != "nan":
        question_text = item["context"] + "\n" + item["question"]
    else:
        question_text = item["question"]
    
    # Combine prompt and question content
    return f"{olympiad_prompt}\n\n{question_text}"


def get_answer_type_text(answer_type, multiple_answer):
    """Helper function to generate appropriate answer type guidance text"""
    # 'Tuple' has various meanings in different context, such as position or values of a series of variable,
    # so it may lead to confusion to directly use 'tuple' in the prompt.
    if not answer_type or ("Need_human_evaluate" in answer_type) or ("Tuple" in answer_type):
        return ""
    
    if not multiple_answer:
        # Single answer
        answer_text = get_single_answer_type_text(answer_type)
        return f"The answer of The problem should be {answer_text}. "
    else:
        # Multiple answers
        if "," not in answer_type:  # Same answer type for all answers
            answer_text = get_single_answer_type_text(answer_type)
            return f"The problem has multiple answers, each of them should be {answer_text}. "
        else:
            answer_types = answer_type.split(",")
            answer_types = [get_single_answer_type_text(t) for t in answer_types]
            if len(set(answer_types)) == 1:
                answer_text = answer_types[0]
                return f"The problem has multiple answers, each of them should be {answer_text}. "
            else:
                answer_text = ", ".join(answer_types)
                return f"The problem has multiple answers, with the answers in order being {answer_text}. "


def get_single_answer_type_text(answer_type):
    """Helper function to get the appropriate description for a single answer type"""
    # Dictionary mapping answer types to descriptions
    english_answer_type_dict = {
        "Numerical": "a numerical value",
        "Expression": "an expression",
        "Equation": "an equation",
        "Interval": "an interval",
    }
    
    if "-" in answer_type:  # No need now
        answer_type = answer_type[: answer_type.find("-")]
    
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            return english_answer_type_dict[t]
    
    # If no match is found (original code would exit with an error)
    return "an answer"  # Graceful fallback

def format_prompt_with_images(item: Dict, images: List, system_prompt: str):
    """Helper function to create properly formatted prompts with images"""
    # For a single image
    if len(images) == 1:
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{system_prompt} {item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        image_data = {"image": images[0]}
    # For multiple images
    else:
        vision_tags = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in range(len(images)))
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{vision_tags}{system_prompt} {item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        image_data = {"image": images}
    
    return prompt_text, image_data, len(images)


def format_text_only_prompt(item: Dict, system_prompt: str):
    """Helper function to create text-only prompts"""
    prompt_text = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{system_prompt} {item['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt_text


def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Special handling for OlympiadBench dataset
        if dataset_name.lower() == "olympiadbench":
            # All OlympiadBench samples have image_path as a list
            if not isinstance(item["image_path"], list):
                continue  # Skip if somehow not a list (shouldn't happen)
                
            # Text-only case
            if item["image_path"] == ["text-only"]:
                prompt_text = format_text_only_prompt_olympiad(item, args.system_prompt)
                
                prompts.append({
                    "prompt": prompt_text,
                })
                
                metadata.append({
                    "dataset": dataset_name,
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "prompt": prompt_text,
                    **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
                })
                continue
                
            # Case with image paths (non-text-only)
            else:
                # Skip if no images are available
                if not item["image_path"]:
                    continue
                    
                # Load all images
                images = []
                for img_path in item["image_path"]:
                    if not os.path.exists(img_path):
                        continue
                    
                    image = load_image(img_path, args.min_pixels, args.max_pixels)
                    if image is not None:
                        images.append(image)
                
                # Skip if no valid images were loaded
                if not images:
                    continue
                
                prompt_text, image_data, num_images = format_prompt_with_images_olympiad(
                    item, images, args.system_prompt
                )
                
                prompts.append({
                    "prompt": prompt_text,
                    "multi_modal_data": image_data,
                })
                
                metadata.append({
                    "dataset": dataset_name,
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "prompt": prompt_text,
                    "num_images": num_images,
                    **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
                })
                continue
        
        # Other datasets (single image path as string)
        if not os.path.exists(item["image_path"]):
            continue
        
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # Use the same text-based formatting for consistency
        prompt_text, image_data, num_images = format_prompt_with_images(
            item, [image], args.system_prompt
        )
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": image_data,
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            "num_images": num_images,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def evaluate_prediction(prediction: str, answer: str, dataset: str, question: str = "", answer_type: str = "") -> float:
    """Evaluate a prediction against the ground truth"""
    if dataset == "geo3k":
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if grade_answer(extracted_answer, answer) else 0.0
    
    # elif dataset == "olympiadbench":
    #     try:
    #         result = boxed_reward_fn(prediction, answer)
    #         return 1.0 if result else 0.0
    #     except Exception:
    #         return 0.0
    
    elif dataset == "dynamath":
        if answer_type == "float":
            extracted_answer = extract_boxed_content(prediction)
            try:
                extracted_answer = float(extracted_answer)
                answer = float(answer)
                return 1.0 if abs(extracted_answer - answer) <= 0.001 else 0.0
            except ValueError:
                return 0.0
            
        else:
            try:
                score = llm_eval_score(question, prediction, answer, dataset)
            except:
                import time
                time.sleep(10)
                score = llm_eval_score(question, prediction, answer, dataset)
                
            return score
    
    elif dataset == "mathvista" or dataset == "mathverse" or dataset == "mathvision" or dataset == "wemath" or dataset == "olympiadbench":
        try:
            score = llm_eval_score(question, prediction, answer, dataset)
        except:
            import time
            time.sleep(10)
            score = llm_eval_score(question, prediction, answer, dataset)
        return score
    
    elif dataset == "logicvista":
        try:
            score = llm_eval_score(question, prediction, answer, dataset)
        except:
            import time
            time.sleep(10)
            score = llm_eval_score(question, prediction, answer, dataset)
        return score

    if dataset == "hallubench":
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if extracted_answer.lower() == answer else 0.0
        # return 1.0 if answer.lower() in prediction.lower() else 0.0
    
    else:
        # Default evaluation
        return 1.0 if extracted_answer == answer else 0.0

def process_outputs(outputs, metadata, max_workers: int) -> Dict[str, List[Dict]]:
    """Process model outputs and calculate metrics"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            meta = metadata[i]
            dataset = meta["dataset"]
            if "question_for_eval" in meta:
                question = meta["question_for_eval"]
            else:
                question = meta["question"]
                
            answer_type = meta.get("answer_type", "text")
            
            future = executor.submit(
                evaluate_prediction, 
                prediction, 
                meta["answer"], 
                dataset,
                question,
                answer_type
            )
            futures.append((future, i, prediction, meta))
        
        for future, i, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
                
                result = {
                    "id": meta["id"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "correct": accuracy > 0,
                    **{k: v for k, v in meta.items() if k not in ["dataset", "id", "question", "answer"]}
                }
                
                results.append(result)
            except Exception as e:
                print(f"Error evaluating prediction {i}: {str(e)}")
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    if not results:
        return {"accuracy": 0.0}
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    metrics = {"accuracy": accuracy}
    
    # Calculate task-specific accuracies if available
    if any("task" in r for r in results):
        task_results = {}
        for r in results:
            if "task" in r:
                task = r["task"]
                if task not in task_results:
                    task_results[task] = []
                task_results[task].append(r["correct"])
        
        task_accuracies = {task: sum(results) / len(results) for task, results in task_results.items()}
        metrics["sub_accuracies"] = task_accuracies
    
    # Calculate problem version accuracies if available
    if any("problem_version" in r for r in results):
        version_results = {}
        for r in results:
            if "problem_version" in r:
                version = r["problem_version"]
                if version not in version_results:
                    version_results[version] = []
                version_results[version].append(r["correct"])
        
        version_accuracies = {version: sum(results) / len(results) for version, results in version_results.items()}
        metrics["sub_accuracies"] = version_accuracies
    
    # Calculate subject accuracies if available
    if any("subject" in r for r in results):
        subject_results = {}
        for r in results:
            if "subject" in r:
                subject = r["subject"]
                if subject not in subject_results:
                    subject_results[subject] = []
                subject_results[subject].append(r["correct"])
        
        subject_accuracies = {subject: sum(results) / len(results) for subject, results in subject_results.items()}
        metrics["sub_accuracies"] = subject_accuracies
        
    if any("source" in r for r in results):
        source_results = {}
        for r in results:
            if "source" in r:
                source = r["source"]
                if source not in source_results:
                    source_results[source] = []
                source_results[source].append(r["correct"])
        
        source_accuracies = {source: sum(results) / len(results) for source, results in source_results.items()}
        metrics["sub_accuracies"] = source_accuracies
        
    if any("level" in r for r in results):
        level_results = {}
        for r in results:
            if "level" in r:
                level = r["level"]
                if level not in level_results:
                    level_results[level] = []
                level_results[level].append(r["correct"])
        
        level_accuracies = {level: sum(results) / len(results) for level, results in level_results.items()}
        metrics["sub_accuracies"] = level_accuracies
    
    return metrics