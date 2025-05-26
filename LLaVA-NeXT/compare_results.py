import json
import sys

# Load base model results
with open('base_evaluation_results.json', 'r') as f:
    base_results = json.load(f)

# Load fine-tuned model results
with open('finetuned_evaluation_results.json', 'r') as f:
    finetuned_results = json.load(f)

# Calculate improvements
improvements = {}
for metric in base_results['metrics']:
    base_value = base_results['metrics'][metric]
    finetuned_value = finetuned_results['metrics'][metric]
    
    # Calculate absolute and relative improvements
    abs_improvement = finetuned_value - base_value
    rel_improvement = (abs_improvement / base_value) * 100 if base_value != 0 else float('inf')
    
    improvements[metric] = {
        "base": base_value,
        "finetuned": finetuned_value,
        "absolute_improvement": abs_improvement,
        "relative_improvement_percent": rel_improvement
    }

# Print comparison table
print("\n===== Model Performance Comparison =====")
print(f"Base model: {base_results['model_path']}")
print(f"Fine-tuned model: {finetuned_results['model_path']} + {finetuned_results['lora_path']}")
print("\n{:<10} {:<10} {:<10} {:<15} {:<10}".format("Metric", "Base", "Finetuned", "Abs. Diff", "Rel. Diff (%)"))
print("-" * 60)

for metric, values in improvements.items():
    print("{:<10} {:<10.4f} {:<10.4f} {:<15.4f} {:<10.2f}".format(
        metric,
        values["base"],
        values["finetuned"],
        values["absolute_improvement"],
        values["relative_improvement_percent"]
    ))

# Save comparison to file
comparison_results = {
    "base_model": base_results['model_path'],
    "finetuned_model": finetuned_results['model_path'],
    "lora_adapter": finetuned_results['lora_path'],
    "metrics_comparison": improvements,
    "sample_count": base_results["Number of samples"]
}

with open('metrics_comparison.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print("\nComparison saved to metrics_comparison.json")
