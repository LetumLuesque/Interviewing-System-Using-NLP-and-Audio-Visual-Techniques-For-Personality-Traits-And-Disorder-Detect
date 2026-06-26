import glob
import os

try:
    list_of_files = glob.glob('logs/*.log')
    if not list_of_files:
        print("No log files found.")
        exit()
        
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"Reading: {latest_file}")

    with open(latest_file, 'r') as f:
        lines = f.readlines()
        
    # Filter for interesting lines
    interesting = []
    for line in lines:
        if 'val_deception_balanced_accuracy' in line or 'val_deception_accuracy' in line or 'val_loss' in line:
            interesting.append(line.strip())
            
    print("\n--- Latest Validation Metrics ---")
    for l in interesting[-20:]:  # Last 20 lines
        print(l)
        
except Exception as e:
    print(f"Error: {e}")
