from pathlib import Path
import sys
from pytact.data_reader import data_reader
import random
import math

dataset_path = Path(sys.argv[1]).resolve()


# after this we will split the training packages into validation and training.
with open("training.txt", "r") as tr:

    train_packages = tr.readlines()
    train_packages = [k.strip() for k in train_packages]

with open("testing_txt.txt", "r") as tst:

    test_packages = tst.readlines()
    test_packages = [k.strip() for k in test_packages]
print(train_packages)

training_data = []
validation_data = []
test_data = []

trainval_definitions = []

# make a list of definitions for training - validation split;
with data_reader(dataset_path) as data:

    for datafile in data.values():

        package_name = datafile.filename.parts[0]
        if package_name in train_packages:

            for d in datafile.definitions():
                trainval_definitions.append(d.name)


print(len(trainval_definitions))
random.seed(17)
random.shuffle(trainval_definitions)

train_end_index = int(math.floor(0.99*len(trainval_definitions)))

train_definitions = set(trainval_definitions[:train_end_index]) # sets for reasonable membership test
val_definitions = set(trainval_definitions[train_end_index:])


with data_reader(dataset_path) as data:
    for datafile in data.values():
        
        package_name =  datafile.filename.parts[0]
                
        for d in datafile.definitions():

            defname = d.name
            if proof := d.proof:
                
                for step in proof:
                    if step.tactic is not None:
                        for outcome in step.outcomes:
                                                    
                            example = f"{outcome.before.text.lstrip()} OUTPUT {outcome.tactic.text_non_anonymous} <END>"
                            
                            if package_name in train_packages:

                                if defname in train_definitions:
                                    training_data.append(example)
                                elif defname in val_definitions:
                                    validation_data.append(example)

                            elif package_name in test_packages:
                                test_data.append(example)

                            
print(len(training_data), len(validation_data), len(test_data))

random.seed(17)

random.shuffle(training_data)
random.shuffle(validation_data)
random.shuffle(test_data)

with open("data/v15_partial_lemmavalsplit_training.txt", "w") as f:
    for line in training_data:
        f.write(line)
        f.write("\n")

with open("data/v15_partial_lemmavalsplit_validation.txt", "w") as f1:
    for line in validation_data:

        f1.write(line)
        f1.write("\n")

with open("data/v15_partial_lemmavalsplit_test.txt", "w") as f2:

    for line in test_data:
        f2.write(line)
        f2.write("\n")

    



