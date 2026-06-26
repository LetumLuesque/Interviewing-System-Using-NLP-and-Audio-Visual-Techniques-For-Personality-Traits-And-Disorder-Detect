import os
import re
import pandas as pd
import glob

def parse_logs(log_dir='logs'):
    """Parse all training logs and stitching them into a single timeline"""
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))
    
    data = []
    
    # Regex patterns
    epoch_pat = re.compile(r'Epoch (\d+)/(\d+)')
    metrics_pat = re.compile(r'training - INFO - \s+([a-zA-Z0-9_]+): ([0-9.]+)')
    stage_pat = re.compile(r'STAGE (\d+):')
    
    current_stage = 1
    
    print(f"Found {len(log_files)} log files. Parsing...")
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        file_data = {} # store metrics for current epoch
        current_epoch = 0
        
        for line in lines:
            # Detect Stage
            stage_match = stage_pat.search(line)
            if stage_match:
                stage = int(stage_match.group(1))
                if stage != current_stage:
                    current_stage = stage
            
            # Detect Epoch Start
            epoch_match = epoch_pat.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Detect Metrics
            if 'val_' in line and 'INFO' in line:
                metric_match = metrics_pat.search(line)
                if metric_match:
                    key = metric_match.group(1)
                    val = float(metric_match.group(2))
                    
                    if current_epoch not in file_data:
                        file_data[current_epoch] = {'stage': current_stage}
                    
                    file_data[current_epoch][key] = val

        # Flatten to list
        for ep, metrics in file_data.items():
            metrics['epoch'] = ep
            metrics['log_file'] = os.path.basename(log_file)
            data.append(metrics)

    df = pd.DataFrame(data)
    
    # Clean up and sort
    if not df.empty:
        df = df.sort_values(by=['stage', 'epoch']).reset_index(drop=True)
        # Create a continuous "Global Epoch" axis
        df['global_step'] = df.index + 1
    
    return df

if __name__ == "__main__":
    print("Exporting Training History...")
    os.makedirs('plots', exist_ok=True)
    
    df = parse_logs()
    
    if not df.empty:
        print(f"Parsed {len(df)} epochs of data.")
        output_path = 'plots/training_history.csv'
        df.to_csv(output_path, index=False)
        print(f"SUCCESS: Data saved to {output_path}")
        print("You can open this CSV in Excel/Sheets to plot your curves.")
    else:
        print("No data found in logs!")
