import pprint

from hafnia.data import load_dataset

dataset = load_dataset("midwest-vehicle-detection")

# Print information on each dataset split
print(dataset)

# View features of the train split
pprint.pprint(dataset["train"].features)

# Print sample from the training set
pprint.pprint(dataset["train"][0])
