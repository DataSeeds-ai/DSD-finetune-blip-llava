from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import warnings
import os
warnings.filterwarnings("ignore")

# Clear CUDA cache
torch.cuda.empty_cache()

# Model paths - using just the base model with vision tower
base_model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
device = "cuda"
device_map = "auto"

# Load image
image_path = "/home/sajjad/AI_proj/example_data/images/celeba/000004.jpg"
image = Image.open(image_path).convert('RGB')

print("Loading pre-trained model...")

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
    customized_config=custom_config,
    multimodal=True
)

print("Using base model without fine-tuning")
model.eval()

# Process image
print("Processing image...")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

# Prepare prompt
conv_template = "qwen_1_5"
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the person's appearance in detail."
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
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=128,
            num_beams=1,
            repetition_penalty=1.2
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