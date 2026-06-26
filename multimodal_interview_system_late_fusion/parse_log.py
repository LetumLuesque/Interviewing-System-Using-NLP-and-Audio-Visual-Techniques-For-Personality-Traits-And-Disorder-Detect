import glob
import os
import re

# Find latest log
list_of_files = glob.glob('logs/training_*.log') 
if not list_of_files:
    print("No log files found")
    exit()
    
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Parsing log: {latest_file}")

with open(latest_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

found_metrics = False
for i in range(len(lines)-1, -1, -1):
    if "Metrics:" in lines[i]:
        print(f"--- Found Metrics at line {i+1} ---")
        # Print next 50 lines
        for j in range(i, min(i+50, len(lines))):
             print(lines[j].strip())
        found_metrics = True
        break

if not found_metrics:
    print("No 'Metrics:' found in log.")
