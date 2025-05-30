import os

dataset_path = "./data/wikiart"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if "_not_detected_" in file:
            new_file = file.replace("_not_detected_", "_not-detected-")
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            print(f"file {file} renamed {new_file}")
print('run over')
