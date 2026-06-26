import traceback
try:
    print("Attempting to import unsloth...")
    import unsloth
    print("Successfully imported unsloth!")
except Exception as e:
    print("Error during import:")
    traceback.print_exc()
