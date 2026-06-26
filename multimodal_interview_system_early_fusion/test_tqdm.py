
import sys
import time
from tqdm import tqdm

class TqdmFileWrapper:
    def __init__(self, file):
        self.file = file
    def write(self, x):
        if len(x.strip()) > 0:
            try:
                self.file.write(x)
                self.file.flush()
            except (OSError, UnicodeEncodeError):
                pass
    def flush(self):
        try:
            self.file.flush()
        except OSError:
            pass

print("Starting Tqdm Test...")
sys.stdout.flush()

try:
    pbar = tqdm(range(20), file=TqdmFileWrapper(sys.stdout), 
               desc="Testing", dynamic_ncols=True, mininterval=0.1)
    for i in pbar:
        time.sleep(0.1)
    print("\nCompleted loop.")
except Exception as e:
    print(f"\nFailed: {e}")

print("End of Test.")
