import logging
import os
import PIL.Image
import io
import time
import re
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
import base64

from openai import OpenAI
import threading

# System messages for different operations
SYSTEM_MESSAGES = {
    "evolve": "You are an excellent vision-language AI assistant that generates more challenging questions while preserving the original answer.",
    "noise": "You are an excellent vision-language AI assistant that adds strategic distractors to questions to make them more challenging while preserving the original answer.",
    "solver": "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.",
    "judge": "You are an expert evaluator that assesses the quality and validity of evolved questions.",
    "noise_judge": "You are an expert evaluator that assesses the effectiveness of noise added to questions and confirms that the original answer remains valid."
}

# def get_evolve_prompt(question: str) -> str:
#     """
#     Generate a prompt for evolving a question into a more challenging version
#     while keeping the original answer.
#     """
#     prompt = f"""
# Transform the following question into a significantly more challenging version that requires deeper reasoning but leads to the same exact answer as the original question.

# **Original Question:**
# {question}

# **Key Requirements:**

# 1. INCREASE COGNITIVE DIFFICULTY - The evolved question MUST require more reasoning steps and deeper analytical thinking to solve.

# 2. PRESERVE EXACT ANSWER - The final answer must be identical to the original question's answer.

# 3. MAINTAIN IMAGE DEPENDENCE - If the original question references visual elements:
#    - AVOID: Converting visual references into explicit measurements
#    - PREFER: "Using the triangle shown in the image..." over explicit values

# 4. INCREASE REASONING DEPTH through:
#    - Requiring multi-step deduction or inference chains
#    - Adding conceptual relationships that must be discovered
#    - Introducing relevant constraints requiring analysis
#    - Restructuring to demand deeper understanding of principles
#    - Creating scenarios where key information must be derived rather than directly used

# 5. AVOID SUPERFICIAL COMPLEXITY:
#    - Do NOT add irrelevant information or noise
#    - Do NOT use unnecessarily complex language
#    - Do NOT introduce ambiguity that permits multiple interpretations

# Your Response Format:
# 1. Complexity Analysis: [Explain specifically how your version increases reasoning difficulty]
# 2. Answer Preservation: [Explain how your version maintains exactly the same answer]
# 3. Evolved Question: [Your transformed question]
# """
#     return prompt.strip()

def get_evolve_prompt(question: str) -> str:
    """
    Generate a prompt for evolving a question into a more challenging version
    while keeping the original answer.
    """
    prompt = f"""
Given an image and the following question, transform it into a significantly more challenging version that requires deeper reasoning but maintains the same answer.

Original Question:
{question}

Your Response Format:
New Question: [Your transformed question]
"""
    return prompt.strip()

def judge_evolved_question_quality(
    original_question: str,
    evolved_question: str,
    image,
    model_name: str = "gemini-1.5-flash-002"
) -> Dict:
    """
    Judge the quality of an evolved question compared to the original.
    Provides a single overall score with explanation.
    """
    prompt = f"""
Evaluate how effectively the evolved question increases difficulty.

Original Question: {original_question}
Evolved Question: {evolved_question}

Examine the provided image carefully and evaluate these criteria, prioritizing them in order of importance:

1. REASONING COMPLEXITY (0-5 points):
   - Does the evolved question require MORE reasoning steps than the original?
   - Does it demand deeper analytical thinking or stronger conceptual understanding?
   - Are additional reasoning steps or inferences necessary?
   
2. QUALITY OF COMPLEXITY (0-3 points):
   - Is the increased difficulty achieved through meaningful reasoning demands rather than superficial complexity?
   - Does it avoid simply adding noise, irrelevant information, or unnecessarily complex language?

3. CLARITY AND FORMULATION (0-2 point):
   - Is the evolved question clearly articulated?
   - Does it properly reference the image rather than pre-extracting visual information?
 
Provide a substantive analysis of each criterion, with particular focus on exactly HOW the question increases reasoning difficulty.

Focus your analysis on how the question requires the image to solve and increases reasoning difficulty.

Respond using exactly this format:
EXPLANATION: [Your detailed explanation here]
SCORE: [single number from 0-10]
"""

    try:
        # Call API to judge question quality with system prompt
        system_instruction = SYSTEM_MESSAGES['judge']
        
        response = call_gemini_api(
            content=[image, "\n", prompt],
            model_name=model_name,
            temperature=0.0,  # Use 0 temperature for consistent evaluation
            max_tokens=1024,
            retries=2,
            system_instruction=system_instruction
        )
        
        # Extract explanation and score from response
        explanation = ""
        score = 0

        explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=SCORE:|$)', response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        # Updated pattern to match both "SCORE: 10" and "SCORE: [10]"
        score_match = re.search(r'SCORE:\s*(?:\[)?(\d+(?:\.\d+)?)(?:\])?', response)
        if score_match:
            try:
                score = int(float(score_match.group(1)))
                # Ensure score is within 0-10 range
                score = max(0, min(10, score))
            except ValueError:
                logging.warning(f"Could not parse score from: {score_match.group(1)}")
                score = 0
        
        if not explanation or score == 0:
            logging.warning(f"Failed to extract proper evaluation from response: {response}")
        
        return {
            "explanation": explanation,
            "score": score,
            "full_response": response
        }
            
    except Exception as e:
        logging.error(f"Error during judge API call: {e}")
        return {
            "explanation": f"Evaluation failed: {str(e)}",
            "score": 0,
            "full_response": str(e)
        }
        
def get_noisy_prompt(question: str) -> str:
    """
    Generate a prompt for adding noise to a question to make it more challenging
    while keeping the original answer and conditions intact.
    """
    prompt = f"""
Transform the following question by adding noise (distractors) to make it more challenging
while ensuring the exact same answer remains correct.

**Original Question:**
{question}

**Key Requirements:**

1. ADD STRATEGIC DISTRACTORS - Include information that:
   - Presents plausible alternative paths that would lead to incorrect answers
   - Introduces tempting but ultimately irrelevant data points
   - Creates decision points where solvers must distinguish relevant from irrelevant information
   - IMPORTANT: These distractors must NOT change the correctness of the original condition

2. PRESERVE EXACT ANSWER AND SOLUTION PATH - The core solution method and final answer 
   must remain identical to the original question. The added noise serves ONLY as distraction.

3. MAINTAIN IMAGE DEPENDENCE - If the original question references visual elements:
   - Continue to refer to the visual elements as shown in the image
   - Add references to visually prominent but irrelevant features
   - The original visual relationships crucial to the solution must remain unaltered

4. MAINTAIN CONCISENESS:
   - Do NOT simply make the question verbose with unnecessary text
   - Each added element should function purely as noise/distractor
   - Aim for cognitive complexity through strategic distraction, not textual complexity

5. NOISE INTEGRATION:
   - Ideally, the noise should blend seamlessly with the core problem and look relevant
   - If it's difficult to add seamless noise while keeping the original conditions correct and preserving the answer, you can add less seamless noise
   - The most important requirement is that the noise does NOT change the correctness of the original condition

6. MAINTAIN A SINGLE QUESTION:
   - The noisy question should still be asking exactly ONE question without any sub-questions or process questions
   - The core question intent must remain singular and focused, despite the added noise
   - Do NOT introduce steps or additional questions that need to be answered

Your Response Format:
1. Noise Analysis: [Explain the specific distractors you've added and how they function as noise]
2. Answer Preservation: [Confirm that filtering out the noise leads to the exact same answer]
3. Noisy Question: [Your transformed question with noise]
"""
    return prompt.strip()

def judge_noisy_question_quality(
    original_question: str,
    noisy_question: str,
    correct_answer: str,
    image,
    model_name: str = "gemini-1.5-flash-002"
) -> Dict:
    """
    Judge the quality of a noisy question compared to the original.
    """
    prompt = f"""
Evaluate how effectively the noisy question increases difficulty while preserving the same answer.

Original Question: {original_question}
Correct Answer: {correct_answer}
Noisy Question: {noisy_question}

Examine the provided image carefully and evaluate these criteria:

1. NOISE EFFECTIVENESS (0-4 points):
   - How well do the distractors create plausible alternative paths?
   - Do they present tempting but incorrect approaches to solving the problem?
   - Do they function purely as noise without affecting the original conditions?
   
2. ANSWER PRESERVATION (0-2 points):
   - Would both questions lead to exactly the same answer: "{correct_answer}"?
   - Is the original solution path still valid after filtering out the noise?
   
3. DISTRACTOR QUALITY (0-2 points):
   - Are the distractors strategically placed to maximize cognitive challenge?
   - Does each added element serve a purpose as meaningful noise rather than just adding verbosity?

4. ONE QUESTION FOCUS  (0-2 point):
   - Despite the noise, does the question still ask exactly ONE clear question?
   - Does it avoid introducing separate, additional questions that need answers?
   - Is the ultimate goal of the question unchanged from the original?
 
Provide a substantive analysis of each criterion, with particular focus on how the noise creates challenge
without altering the original conditions or correct answer.

Respond using exactly this format:
EXPLANATION: [Your detailed explanation here]
SCORE: [single number from 0-10]
"""
    try:
        # Call API to judge question quality with system prompt
        system_instruction = SYSTEM_MESSAGES['noise_judge']
        
        response = call_gemini_api(
            content=[image, "\n", prompt],
            model_name=model_name,
            temperature=0.0,  # Use 0 temperature for consistent evaluation
            max_tokens=1024,
            retries=2,
            system_instruction=system_instruction
        )
        
        # Extract explanation and score from response
        explanation = ""
        score = 0
        
        explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=SCORE:|$)', response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response)
        if score_match:
            try:
                score = int(float(score_match.group(1)))
                # Ensure score is within 0-10 range
                score = max(0, min(10, score))
            except ValueError:
                logging.warning(f"Could not parse score from: {score_match.group(1)}")
                score = 0
        
        if not explanation or score == 0:
            logging.warning(f"Failed to extract proper evaluation from response: {response[:100]}...")
        
        return {
            "explanation": explanation,
            "score": score,
            "full_response": response
        }
            
    except Exception as e:
        logging.error(f"Error during judge API call: {e}")
        return {
            "explanation": f"Evaluation failed: {str(e)}",
            "score": 0,
            "full_response": str(e)
        }

def get_solver_prompt(question: str) -> str:
    """
    Generate a streamlined prompt for solving a question based on an image.
    """
    prompt = f"{question}"
    return prompt.strip()

def call_gemini_api(content, model_name="gemini-1.5-flash-002", temperature=0.5, max_tokens=512, retries=2, system_instruction=None):
    """
    Modified to handle PIL Images properly for Gemini API
    """
    model = genai.GenerativeModel(model_name=model_name)
    
    if system_instruction is None:
        system_instruction = "You are a helpful assistant."
    
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
        
# Create thread-local storage for client instances
thread_local = threading.local()

# Semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(64)  # Adjust to appropriate value

def get_client():
    """Get thread-local OpenAI client instance"""
    if not hasattr(thread_local, "client"):
        qwen_api_key = os.getenv("SILICON_FLOW_API_KEY")
        if not qwen_api_key:
            raise ValueError("API_KEY environment variable is not set")
        
        thread_local.client = OpenAI(
            api_key=qwen_api_key,
            base_url="https://api.siliconflow.cn/v1",
            # Add connection pool configuration
            timeout=60.0,  # Total timeout
            max_retries=0,  # We handle retries manually
        )
    return thread_local.client

def call_qwen_api(content, model_name="qwen2.5-vl-7b-instruct", temperature=0.5, max_tokens=512, retries=2, system_instruction=None):
    """
    Call Qwen API with proper formatting for images and messages
    
    Args:
        content: List of content items (text or images)
        model_name: Name of the Qwen model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        retries: Number of retries on failure
        system_instruction: Optional system instruction
    """
 
    # Format messages for OpenAI-compatible API
    messages = []
    
    # Keeping original system_instruction logic
    if not system_instruction:
        system_instruction = "You are a helpful assistant."
    
    # Add system instruction if provided
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    current_message = {"role": "user", "content": []}
    
    # Convert content list to proper OpenAI format
    for item in content:
        if isinstance(item, PIL.Image.Image):
            # Convert PIL Image to base64
            with io.BytesIO() as bio:
                item.save(bio, format='PNG')
                img_bytes = bio.getvalue()
            
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            
            current_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        else:
            # Add text content
            current_message["content"].append({
                "type": "text",
                "text": item
            })
    
    messages.append(current_message)
    
    # Use semaphore to limit concurrent API calls
    with api_semaphore:
        start_time = time.time()
        for attempt in range(retries + 1):
            try:
                # Get thread-local client
                client = get_client()
                
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                elapsed = time.time() - start_time
                logging.debug(f"API call successful after {elapsed:.2f}s (attempt {attempt + 1})")
                
                # Extract and return the response text
                return completion.choices[0].message.content
                
            except Exception as e:
                elapsed = time.time() - start_time
                logging.error(f"API call error after {elapsed:.2f}s (attempt {attempt + 1}/{retries + 1}): {str(e)}")
                if attempt < retries:
                    # Exponential backoff, but wait at most 30 seconds
                    sleep_time = min(2 ** attempt, 30)
                    logging.warning(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                raise

def extract_evolved_question(text: str) -> Optional[str]:
    """
    Extracts the new question from the model's response with improved robustness.
    """
    # Remove asterisks and normalize line breaks
    clean_text = text.replace('*', '').replace('\r\n', '\n')
    
    # Check for various possible formats of the marker
    markers = ["New Question:", "new question:", "New question:", "NEW QUESTION:"]
    
    for marker in markers:
        if marker in clean_text:
            # Split by the marker and take what comes after
            parts = clean_text.split(marker, 1)
            if len(parts) > 1:
                # Get the content after the marker and before double newline
                content = parts[1].strip()
                if "\n\n" in content:
                    content = content.split("\n\n", 1)[0].strip()
                return content
    
    return None

def extract_noisy_question(text: str) -> Optional[str]:
    """
    Extracts the evolved question from the model's response using simple string matching,
    without checking for end markers.
    """
    # Remove asterisks from the text to handle both with and without formatting
    clean_text = text.replace('*', '')
    
    # Check if "Evolved Question:" appears in the cleaned text
    if "Noisy Question:" in clean_text:
        # Split by the marker and take what comes after
        parts = clean_text.split("Noisy Question:", 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the final answer from the model's response that appears inside \boxed{} tags.
    Uses string operations instead of regex to avoid escape sequence problems.
    """
    if not isinstance(text, str):
        return None
        
    try:
        # Look for \boxed{} pattern
        box_start = text.find("\\boxed{")
        
        if box_start != -1:
            # Find the starting position after \boxed{
            content_start = box_start + len("\\boxed{")
            
            # Find the matching closing brace
            brace_count = 1
            content_end = content_start
            
            while content_end < len(text) and brace_count > 0:
                if text[content_end] == '{':
                    brace_count += 1
                elif text[content_end] == '}':
                    brace_count -= 1
                content_end += 1
            
            # If we found a matching closing brace, extract the content
            if brace_count == 0:
                # Adjust position to exclude the closing brace
                content_end -= 1
                return text[content_start:content_end].strip()
        
        # Fallback: if no \boxed{} format is found, return the last paragraph
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1].strip()
        
        return None
    except Exception as e:
        logging.error(f"Error in extract_final_answer: {e}")
        return "Error extracting answer"

def verify_answers_match(
    original_answer: str,
    generated_answer: str,
    model_name: str = "gemini-1.5-flash-002"
) -> bool:
    """
    Use an LLM to verify if two answers are equivalent, without requiring image input.
    Returns a simple True/False verdict.
    """
    prompt = f"""
Compare these two answers and determine if they are semantically equivalent (represent the same correct answer).

Original Answer: {original_answer}

Generated Answer: {generated_answer}

Carefully analyze both answers, considering:
- Mathematical equivalence (e.g., "1/2" equals "0.5")
- Different ways of expressing the same measurement (e.g., "30 degrees" equals "π/6 radians")
- Equivalent expressions (e.g., "the area equals 36 square units" equals "36 square units")
- Different but correct ways of expressing the same concept
- If one answer provides a number without units and the other includes units, they can be considered equivalent if the numerical value is correct

Only respond with "TRUE" if they are equivalent answers to the same question, or "FALSE" if they represent different answers.
"""
    
    try:
        # Call API to verify answer equivalence (no system prompt needed for simple task)
        response = call_gemini_api(
            content=[prompt],
            model_name=model_name,
            temperature=0.0,  # Use 0 temperature for consistent verification
            max_tokens=128,
            retries=2
        )
    
        # Extract the final TRUE/FALSE verdict
        words = response.strip().upper().split()
                
        # Filter to only TRUE or FALSE words
        verdicts = [word for word in words if word == "TRUE" or word == "FALSE"]
                
        if verdicts and verdicts[-1] == "TRUE":
            return True
        else:
            # Return False if last verdict is FALSE or if no verdict found
            return False
            
    except Exception as e:
        logging.error(f"Error during answer verification: {e}")
        return False