import argparse
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import re

LOG_DIRECTORIES = [
    "./LAVIS/lavis/output/BLIP2/Caption_Gurushots_OPT2.7b_Run1/20250415092/",
    "./LAVIS/lavis/output/BLIP2/Caption_Gurushots_OPT2.7b_Base/20250415114/",

]

def parse_log_file(log_path):
    """Parses the log.txt file to extract training and validation metrics."""
    data = {"epoch": [], "train_loss": [], "val_metrics": {}}
    current_epoch = 0
    epoch_train_losses = []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        log_start_index = 0
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('{"train_lr"') or stripped_line.startswith('{"val_'):
                log_start_index = i
                break
        else: 
            print(f"Warning: No standard log entries found in {log_path}")
            config_end_found = False
            brace_level = 0
            for i, line in enumerate(lines):
                if '{' in line:
                    brace_level += line.count('{')
                if '}' in line:
                    brace_level -= line.count('}')
                if brace_level == 0 and line.strip() == '}':
                    log_start_index = i + 1
                    config_end_found = True
                    break
            if not config_end_found:
                 if lines and (lines[0].strip().startswith('{"train_lr"') or lines[0].strip().startswith('{"val_')):
                    log_start_index = 0
                 else:
                    print(f"Error: Could not determine the start of log entries in {log_path}. Please check the file format.")
                    return None # Cannot proceed if we don't know where logs start


        for line in lines[log_start_index:]:
            line = line.strip()
            if not line:
                continue
            try:
                log_entry = json.loads(line)

                if "train_loss" in log_entry:
                    try:
                        epoch_train_losses.append(float(log_entry["train_loss"]))
                    except (ValueError, TypeError):
                        pass

                elif "val_Bleu_1" in log_entry:
                    if epoch_train_losses:
                        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
                        data["epoch"].append(current_epoch)
                        data["train_loss"].append(avg_train_loss)

                        processed_metrics_this_epoch = set()
                        # Store validation metrics found in this log entry
                        for key, value in log_entry.items():
                            if key.startswith("val_"):
                                metric_name = key.replace("val_", "")
                                processed_metrics_this_epoch.add(metric_name)
                                if metric_name not in data["val_metrics"]:
                                    # Initialize with Nones for previous epochs if this is a new metric
                                    data["val_metrics"][metric_name] = [None] * current_epoch
                                
                                # Pad if this metric was missing in some previous epochs
                                if len(data["val_metrics"][metric_name]) < current_epoch:
                                     data["val_metrics"][metric_name].extend([None] * (current_epoch - len(data["val_metrics"][metric_name])))

                                try:
                                    data["val_metrics"][metric_name].append(float(value))
                                except (ValueError, TypeError):
                                    data["val_metrics"][metric_name].append(None)

                        # Pad metrics that were known but *not* in this specific log entry
                        all_known_metrics = set(data["val_metrics"].keys())
                        missing_metrics_this_epoch = all_known_metrics - processed_metrics_this_epoch
                        for metric_name in missing_metrics_this_epoch:
                             # Pad if this metric was missing in some previous epochs
                             if len(data["val_metrics"][metric_name]) < current_epoch:
                                 data["val_metrics"][metric_name].extend([None] * (current_epoch - len(data["val_metrics"][metric_name])))
                             # Append None for the current epoch
                             data["val_metrics"][metric_name].append(None)

                        # Increment epoch and reset train losses for the next one
                        current_epoch += 1
                        epoch_train_losses = []

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {log_path}: {line}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line in {log_path}: {line} - {e}")
                continue

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return None

    # Convert collected data to DataFrame
    df_train = pd.DataFrame({'epoch': data['epoch'], 'train_loss': data['train_loss']})
    df_val = pd.DataFrame(data['val_metrics'])

    # Check if lengths match before assigning epoch column to df_val
    if not df_val.empty:
        if len(data['epoch']) == len(df_val):
            df_val['epoch'] = data['epoch']
        else:
             print(f"Error: Mismatch between epoch count ({len(data['epoch'])}) and validation metric count ({len(df_val)}) after parsing {log_path}. Cannot merge correctly.")
             return None 

    # Handle cases with no val data or no training data
    if df_train.empty and df_val.empty:
         return None
    elif df_val.empty:
         df = df_train
    elif df_train.empty:
         df = df_val
    else:
        df = pd.merge(df_train, df_val, on="epoch", how="outer").sort_values(by="epoch")

    return df

def plot_metrics(df, run_name, output_dir):
    """Generates and saves plots for specified metrics for a single run."""
    if df is None or df.empty:
        print(f"No data to plot for run: {run_name}")
        return

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = {
        "train_loss": "Train Loss",
        "Bleu_4": "Validation BLEU-4",
        "ROUGE_L": "Validation ROUGE-L",
        "agg_metrics": "Validation Aggregate Metrics"
    }

    available_metrics = {k: v for k, v in metrics_to_plot.items() if k in df.columns}

    if not available_metrics:
        print(f"No plottable metrics found in the data for run: {run_name}")
        return

    for metric_key, plot_title in available_metrics.items():
        plt.figure(figsize=(10, 6))
        # Check if the metric column has any non-null data before plotting
        if not df[metric_key].isnull().all():
            plt.plot(df["epoch"], df[metric_key], marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel(plot_title.split()[-1]) # Use metric name as Y-axis label
            plt.title(f"{plot_title} vs. Epoch ({run_name})")
            # No legend needed for a single line plot
            plt.grid(True)
            plot_filename = os.path.join(output_dir, f"{metric_key}_vs_epoch.png")
            try:
                plt.savefig(plot_filename)
                print(f"Saved plot: {plot_filename}")
            except Exception as e:
                print(f"Error saving plot {plot_filename}: {e}")
        else:
             print(f"Skipping plot for {plot_title} ({run_name}) - metric column is all null.")
        plt.close() # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from BLIP2 log files into separate run directories.")
    parser.add_argument("--output-dir", default="plots", help="Base directory to save the plot subdirectories (relative to script location).")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_path = os.path.join(script_dir, args.output_dir)

    processed_any = False
    for log_dir in LOG_DIRECTORIES:
        if not os.path.isdir(log_dir):
             print(f"Warning: Directory not found, skipping: {log_dir}")
             continue

        log_file = os.path.join(log_dir, "log.txt")
        print(f"Processing: {log_file}")
        df = parse_log_file(log_file)

        if df is not None and not df.empty:
            # Extract a meaningful run name (e.g., the last two parts of the path)
            parts = [p for p in log_dir.strip(os.sep).split(os.sep) if p]
            # Sanitize run name to be filesystem friendly
            raw_run_name = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else os.path.basename(log_dir.rstrip(os.sep))
            # Replace characters potentially problematic for directory names
            run_name = re.sub(r'[<>:"/\\|?*]', '_', raw_run_name) 
            if not run_name: # Handle edge case of empty name after sanitization
                run_name = f"run_{log_dir.hash()}" # Or generate a unique ID

            run_output_dir = os.path.join(base_output_path, run_name)
            print(f"Output directory for this run: {run_output_dir}")
            plot_metrics(df, run_name, run_output_dir)
            processed_any = True
        elif df is None:
            print(f"Warning: Failed to parse log file {log_file}")
        else: # df is empty
            print(f"Warning: No data found in {log_file}")


    if not processed_any:
        print("Error: No valid log data found or processed in any of the specified directories.")
        return

    print(f"Plot generation complete. Check the subdirectories inside {base_output_path}")

if __name__ == "__main__":
    main()
