with open(r"C:\Program Files\Python311\Lib\site-packages\unsloth_zoo\__init__.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines[:40]):
        print(f"{idx+1}: {line}", end="")
