#!/bin/bash


# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run evaluation on base model
echo "Evaluating base model..."
python scripts/inf/evaluation_script.py \
    --model-path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
    --eval-data-path "/home/sajjad/AI_proj/example_data/val_data_guru.json" \
    --image-dir "/home/sajjad/AI_proj/example_data" \
    --output-file "base_evaluation_results.json" \
    --detailed-output "base_detailed_results.json"

# Run evaluation on fine-tuned model
echo -e "\nEvaluating fine-tuned model..."
python scripts/inf/evaluation_script.py \
    --model-path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
    --lora-path "/home/sajjad/AI_proj/outputs/llava-onevision-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-0.5b-ov-minimal-adpt_lora16" \
    --eval-data-path "/home/sajjad/AI_proj/example_data/val_data_guru.json" \
    --image-dir "/home/sajjad/AI_proj/example_data" \
    --output-file "finetuned_evaluation_results.json" \
    --detailed-output "finetuned_detailed_results.json"

# Create comparison script
echo -e "\nCreating and running comparison script..."

# Run the comparison script
python compare_results.py

echo -e "\nEvaluation complete. Results saved to base_evaluation_results.json and finetuned_evaluation_results.json"
echo "Comparison results saved to metrics_comparison.json"