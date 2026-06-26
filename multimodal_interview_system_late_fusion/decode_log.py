import glob
import os
import sys

# Find latest log
list_of_files = glob.glob('logs/training_*.log') 
if not list_of_files:
    print("No log files")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"Reading: {latest_file}")

with open(latest_file, 'rb') as f:
    data = f.read()

encodings = ['utf-8', 'utf-16-le', 'cp1252', 'latin1']

for enc in encodings:
    try:
        print(f"\n--- Trying Encoding: {enc} ---")
        text = data.decode(enc)
        lines = text.split('\n')
        metric_indices = [i for i, line in enumerate(lines) if "Epoch" in line and "Metrics" in line]
        loss_indices = [i for i, line in enumerate(lines) if "loss=" in line and "Train" in line]
        
        if loss_indices:
             print(f"Latest Training Loss (Line {loss_indices[-1]}):")
             print(lines[loss_indices[-1]].strip())
        
        if metric_indices:
            print(f"SUCCESS! Found {len(metric_indices)} metric blocks.")
            last_idx = metric_indices[-1]
            print(f"--- Printing Latest Block (Line {last_idx}) ---")
            for j in range(last_idx, min(len(lines), last_idx+20)):
                print(lines[j].strip())
            break
    except Exception as e:
        print(f"Failed: {e}")
