import json
import os
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from PIL import Image
import argparse
import sys

# Import model-related modules from your existing script
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_qwen import LlavaQwenConfig
from peft import PeftModel, PeftConfig
import copy

def load_model(model_path, lora_path=None, device="cuda", device_map="auto"):
    """Load the LLaVA model with optional LoRA adapter."""
    print("Loading pre-trained model...")
    
    # Determine if this is a checkpoint path
    is_checkpoint = False
    if lora_path and os.path.basename(lora_path).startswith("checkpoint-"):
        is_checkpoint = True
        print(f"Detected checkpoint path: {lora_path}")
    
    # For checkpoints, use the original model path
    # For main folder, we can use the provided path directly
    base_model_path = model_path
    
    # Create a custom config with all necessary parameters
    try:
        custom_config = LlavaQwenConfig.from_pretrained(base_model_path)
        print("Loaded custom config for Qwen model")
    except Exception as e:
        print(f"Warning: Could not load custom config: {str(e)}")
        custom_config = None
    
    # Load the model with custom config
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=base_model_path,
        model_base=None,
        model_name="llava_qwen_for_causal_lm",  # Use the causal LM version explicitly
        load_8bit=False,
        load_4bit=False,
        device_map=device_map,
        torch_dtype="bfloat16",
        customized_config=custom_config,
        multimodal=True
    )
    
    print(f"Model type: {type(model)}")
    
    # Apply LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        
        # Determine which adapter file to use
        if os.path.exists(adapter_path):
            print(f"Using adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, lora_path)
        elif os.path.exists(bin_path):
            print(f"Using adapter from: {bin_path}")
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            print(f"Warning: No adapter model found in {lora_path}")
        
        # Load non-LoRA trainable weights if they exist
        non_lora_files = [
            os.path.join(lora_path, "non_lora_trainables.bin"),  # Main folder format
            os.path.join(lora_path, "pytorch_model.bin")         # Sometimes used in checkpoints
        ]
        
        non_lora_loaded = False
        for non_lora_path in non_lora_files:
            if os.path.exists(non_lora_path) and not non_lora_loaded:
                print(f"Loading non-LoRA trainables from {non_lora_path}...")
                try:
                    non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
                    
                    # Filter out LoRA weights (if they exist in this file)
                    non_lora_state_dict = {k: v for k, v in non_lora_state_dict.items() 
                                          if not any(l in k for l in ["lora_A", "lora_B"])}
                    
                    # Get the appropriate model to update (base model for PeftModel)
                    target_model = model.base_model if hasattr(model, "base_model") else model
                    
                    # Try to load the weights
                    missing, unexpected = target_model.load_state_dict(non_lora_state_dict, strict=False)
                    print(f"Loaded non-LoRA weights: {len(missing)} missing and {len(unexpected)} unexpected keys")
                    non_lora_loaded = True
                except Exception as e:
                    print(f"Error loading non-LoRA weights from {non_lora_path}: {str(e)}")
                    continue
        
        if not non_lora_loaded:
            print("No non-LoRA trainable weights were loaded")
    
    model.eval()
    return tokenizer, model, image_processor, max_length

def generate_description(model, tokenizer, image_processor, image_path, use_lora=False, device="cuda"):
    """Generate a description for a given image."""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
        
        # Prepare prompt
        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + "\nDescribe this scene (at least 20 to 50 words and not more than 80 words): focus on the overall context, environment, lighting, camera angle (eye level/high/low/bird's eye), color palette, photography style, and any text visible in the image."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]
        
        # Generate response
        with torch.inference_mode():
            generation_model = model
            
            # For PEFT models, carefully find the model with generate capability
            if use_lora:
                if hasattr(model, "generate"):
                    generation_model = model
                elif hasattr(model, "base_model") and hasattr(model.base_model, "generate"):
                    generation_model = model.base_model
                else:
                    # Try to find the model with generate method through common patterns
                    found = False
                    for target_model in [model, getattr(model, "base_model", None)]:
                        if target_model is None:
                            continue
                            
                        for attr_name in ["model", "transformer", "language_model"]:
                            if hasattr(target_model, attr_name):
                                sub_model = getattr(target_model, attr_name)
                                if hasattr(sub_model, "generate"):
                                    generation_model = sub_model
                                    found = True
                                    break
                        if found:
                            break
            
            #print(f"Generation model type: {type(generation_model)}")
            
            # Different handling for PeftModel vs regular model
            if isinstance(generation_model, PeftModel) or "PeftModel" in str(type(generation_model)):
                # For PeftModel, we need to pass all parameters as kwargs
                output_ids = generation_model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=192,
                    repetition_penalty=1.2
                )
            else:
                # For regular models, we can use positional args
                output_ids = generation_model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=192,
                    repetition_penalty=1.2
                )
        
        # Decode output
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract just the model's response (remove the conversation history)
        response_parts = output_text.split(question)
        if len(response_parts) > 1:
            output_text = response_parts[1].strip()
        
        return output_text
    except Exception as e:
        print(f"Error generating description for {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def compute_bleu(generated_texts, reference_texts):
    """Compute BLEU-4 score."""
    # Tokenize texts
    tokenized_refs = [[ref.lower().split()] for ref in reference_texts]
    tokenized_gens = [gen.lower().split() for gen in generated_texts]
    
    # Use smoothing function to handle cases where there are no n-gram overlaps
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU-4 score
    bleu_score = corpus_bleu(tokenized_refs, tokenized_gens, 
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoothie)
    
    return bleu_score

def compute_rouge(generated_texts, reference_texts):
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        score = scorer.score(gen, ref)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    return np.mean(rouge_scores)

def compute_bertscore(generated_texts, reference_texts):
    """Compute BERTScore."""
    P, R, F1 = bert_score(generated_texts, reference_texts, lang="en", rescale_with_baseline=True)
    return F1.mean().item()

def main(args):
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Load the evaluation data
    with open(args.eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Determine model type and prepare output file names
    if args.lora_path:
        if "checkpoint-" in args.lora_path:
            model_type = f"checkpoint-{os.path.basename(args.lora_path).split('-')[-1]}"
        else:
            model_type = "finetuned"
    else:
        model_type = "base"
    
    output_prefix = f"{model_type}_"
    
    # Update output file paths if not explicitly provided
    if args.output_file == "evaluation_results.json":
        args.output_file = f"{output_prefix}evaluation_results.json"
    if args.detailed_output == "detailed_results.json":
        args.detailed_output = f"{output_prefix}detailed_results.json"
    
    # Load the model (base or with LoRA)
    tokenizer, model, image_processor, _ = load_model(
        args.model_path, 
        args.lora_path, 
        args.device, 
        args.device_map
    )
    
    # Generate descriptions and compute metrics
    generated_descriptions = []
    reference_descriptions = []
    image_paths = []
    
    # Limit to 50 samples for testing if needed
    #eval_limit = args.eval_limit if args.eval_limit > 0 else len(eval_data)
    eval_limit = len(eval_data)
    subset_data = eval_data[0:eval_limit]
    
    print(f"Processing {len(subset_data)} images...")
    for item in tqdm(subset_data):
        # Construct full image path
        image_rel_path = item['image']
        image_path = os.path.join(args.image_dir, image_rel_path)
        
        # Generate description
        generated_desc = generate_description(
            model, 
            tokenizer, 
            image_processor, 
            image_path, 
            use_lora=(args.lora_path is not None),
            device=args.device
        )
        
        # Extract reference description from conversations
        reference_desc = ""
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                reference_desc = conv['value']
                break
        
        if generated_desc and reference_desc:
            generated_descriptions.append(generated_desc)
            reference_descriptions.append(reference_desc)
            image_paths.append(image_rel_path)
    
    # Calculate metrics
    print(f"\nComputing metrics for {len(generated_descriptions)} valid image-description pairs...")
    
    bleu_score = compute_bleu(generated_descriptions, reference_descriptions)
    rouge_score = compute_rouge(generated_descriptions, reference_descriptions)
    bert_score = compute_bertscore(generated_descriptions, reference_descriptions)
    
    # Print and save results
    results = {
        "model_type": model_type,
        "model_path": args.model_path,
        "lora_path": args.lora_path if args.lora_path else "None",
        "metrics": {
            "BLEU-4": bleu_score,
            "ROUGE-L": rouge_score,
            "BERTScore": bert_score,
        },
        "Number of samples": len(generated_descriptions)
    }
    
    print("\nEvaluation Results:")
    print(f"Model Type: {model_type}")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")
    print(f"Number of samples: {len(generated_descriptions)}")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")
    
    # Save all generated descriptions for manual inspection
    output_data = []
    for i, (img, gen, ref) in enumerate(zip(image_paths, generated_descriptions, reference_descriptions)):
        output_data.append({
            "image": img,
            "generated": gen,
            "reference": ref
        })
    
    with open(args.detailed_output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Detailed results saved to {args.detailed_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image description model")
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
                        help="Path to the base model")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to the LoRA adapter (main folder or checkpoint)")
    parser.add_argument("--eval-data-path", type=str, required=True,
                        help="Path to the evaluation data JSON")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device mapping strategy")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--detailed-output", type=str, default="detailed_results.json",
                        help="Path to save detailed evaluation results")
    parser.add_argument("--eval-limit", type=int, default=5,
                        help="Limit evaluation to N samples (default: 50, use 0 for all)")
    
    args = parser.parse_args()
    main(args)