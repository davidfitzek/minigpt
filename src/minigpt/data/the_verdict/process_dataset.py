import os
from minigpt.data.utils import prepare_dataset

script_dir = os.path.join(os.path.dirname(__file__))

# Process the dataset
data_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
result = prepare_dataset(data_url, script_dir)

# Print summary
print("The verdict dataset processed.")
print(f"train.bin has {result['train_tokens']:,} tokens")
print(f"val.bin has {result['val_tokens']:,} tokens")