from datasets import load_dataset

train_dataset = load_dataset("planktons_dataset", "2013-14", split="train")
test_dataset = load_dataset("planktons_dataset", "2013-14", split="test")

print(f"train data size: {len(train_dataset)}")
print(f"test data size: {len(test_dataset)}")

# # pick a random image
# import random

# num = random.randint(0, len(dataset))

# print(dataset.features["label"].names[dataset[num]["label"]])
# dataset[num]['image'].show()
