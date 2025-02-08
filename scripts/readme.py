from datasets import load_dataset

dataset = load_dataset("alexjackson17/countdown-numbers-6-gr")

# Example: Access the first entry in the training split
example = dataset["train"][0]
print("Numbers: ", example["starting"])
print("Target: ", example["target"])
print("Closest: ", example["closest"])
print("Expression: ", example["expression"])
print("Difference: ", example["delta"])
print("Score: ", example["score"])
