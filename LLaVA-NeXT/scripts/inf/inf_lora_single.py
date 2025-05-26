from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from peft import PeftModel
from PIL import Image
import copy
import torch
import warnings
import os
import json
from transformers import AutoConfig
warnings.filterwarnings("ignore")

# Clear CUDA cache
torch.cuda.empty_cache()

# Model paths
lora_path = "/home/sajjad/AI_proj/outputs/llava-onevision-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-0.5b-ov-minimal-adpt"
base_model = 'lmms-lab/llava-onevision-qwen2-0.5b-ov'
device = "cuda"
device_map = "auto"

# Load image
image_path = "/home/sajjad/AI_proj/example_data/images/celeba/000004.jpg"
image = Image.open(image_path).convert('RGB')

# Import the necessary classes to create a custom config
from llava.model.language_model.llava_qwen import LlavaQwenConfig

# Create a custom config with all necessary parameters
custom_config = LlavaQwenConfig.from_pretrained(base_model)

# Load the model with custom config
tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path=base_model,
    model_base=None,
    model_name="llava_qwen",
    load_8bit=False,
    load_4bit=False,
    device_map=device_map,
    torch_dtype="bfloat16",
    multimodal=True
)

# Now apply the LoRA adapter
print(f"Loading LoRA adapter from {lora_path}...")
if os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
    model = PeftModel.from_pretrained(model, lora_path)
    print("LoRA adapter loaded successfully")
    
    # Load non-LoRA trainable weights if they exist
    non_lora_path = os.path.join(lora_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print(f"Loading non-LoRA trainables from {non_lora_path}...")
        non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
        
        # Get the appropriate model to update (base model for PeftModel)
        target_model = model.base_model if hasattr(model, "base_model") else model
        
        # Print some diagnostic information about the model structure
        print("\n=== Model Structure Diagnostics ===")
        if hasattr(target_model, "model"):
            print("target_model has 'model' attribute")
            if hasattr(target_model.model, "model"):
                print("target_model.model has 'model' attribute")
                if hasattr(target_model.model.model, "mm_projector"):
                    print("Found mm_projector at: target_model.model.model.mm_projector")
                    print(f"Type: {type(target_model.model.model.mm_projector)}")
        
        # Also check direct mm_projector
        if hasattr(target_model, "mm_projector"):
            print("Found mm_projector at: target_model.mm_projector")
            print(f"Type: {type(target_model.mm_projector)}")
        
        # Print the first few keys in the non_lora_state_dict
        #print("\n=== Non-LoRA State Dict Keys ===")
        #print(f"First 10 keys: {list(non_lora_state_dict.keys())}")
        
        # Filter for mm_projector related keys
        mm_projector_keys = [k for k in non_lora_state_dict.keys() if "mm_projector" in k]
        print(f"\nFound {len(mm_projector_keys)} mm_projector related keys:")
        for k in mm_projector_keys:
            print(f"  - {k}")
        
        # Based on the diagnostic output, target mm_projector directly
        print("\n=== Loading mm_projector weights ===")
        loaded_successfully = False
        
        if hasattr(target_model, "model") and hasattr(target_model.model, "model") and hasattr(target_model.model.model, "mm_projector"):
            mm_projector = target_model.model.model.mm_projector
            print(f"Targeting mm_projector at: target_model.model.model.mm_projector")
            
            # Create a state dict for the mm_projector based on the keys we found
            mm_projector_dict = {}
            for k, v in non_lora_state_dict.items():
                if "mm_projector" in k:
                    parts = k.split("mm_projector.")
                    if len(parts) > 1:
                        suffix = parts[1]  # e.g., "0.weight"
                        mm_projector_dict[suffix] = v
            
            print(f"Extracted {len(mm_projector_dict)} mm_projector keys from non_lora_trainables")
            print(f"Example mapping: base_model.model.model.mm_projector.0.weight -> 0.weight")
            
            try:
                missing, unexpected = mm_projector.load_state_dict(mm_projector_dict, strict=False)
                print(f"Loaded mm_projector: {len(missing)} missing and {len(unexpected)} unexpected keys")
                
                if len(missing) == 0:
                    print("Successfully loaded all mm_projector weights!")
                    loaded_successfully = True
                else:
                    print("Missing keys:", missing)
            except Exception as e:
                print(f"Error loading mm_projector: {str(e)}")
                
            # Compare the weights - FIXED VERSION with device handling
            if loaded_successfully:
                print("\nVerifying loaded weights match the source weights:")
                for suffix, tensor in mm_projector_dict.items():
                    # Move source tensor to same device as model parameters
                    source_tensor = tensor.to(device)
                    
                    # Get the loaded tensor from the model
                    for name, param in mm_projector.named_parameters():
                        if name == suffix:
                            # Now both tensors are on the same device
                            if torch.allclose(param.data, source_tensor, rtol=1e-3):
                                print(f"✓ Weight {suffix} matches")
                            else:
                                print(f"✗ Weight {suffix} does NOT match")
                                # Print some stats to see differences
                                print(f"  Source mean: {source_tensor.mean().item()}, Model mean: {param.data.mean().item()}")
                                print(f"  Source std: {source_tensor.std().item()}, Model std: {param.data.std().item()}")
            
        if not loaded_successfully:
            print("\nWarning: Could not find an exact match for mm_projector weights.")
            print("Falling back to loading key specific mm_projector weights")
            
            # Try a different approach - create a filtered dictionary with only mm_projector weights
            filtered_dict = {}
            for k, v in non_lora_state_dict.items():
                if "mm_projector" in k:
                    filtered_dict[k] = v
            
            if filtered_dict:
                try:
                    # Load only mm_projector related weights
                    missing, unexpected = target_model.load_state_dict(filtered_dict, strict=False)
                    print(f"Global mm_projector loading: {len(missing)} missing and {len(unexpected)} unexpected keys")
                except Exception as e:
                    print(f"Error in fallback loading: {str(e)}")

model.eval()

# Process image
print("\n=== Processing Image and Generating Response ===")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

# Prepare prompt
conv_template = "qwen_1_5"
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the person's appearance."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print("Prompt:", prompt_question)

# Tokenize
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

try:
    print("Generating response...")
    with torch.inference_mode():
        # For PEFT models, we need to access the base model's generate method
        if hasattr(model, "base_model"):
            output_ids = model.base_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                #temperature=0.7,
                max_new_tokens=192,  # Increased to allow for more detailed output
            )
        else:
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=192,
            )

    # Print results
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("\nImage:", image_path)
    print("Generated description:", output_text)
    print("Word count:", len(output_text.split()))
except Exception as e:
    print(f"Error during generation: {str(e)}")
    import traceback
    traceback.print_exc()