import os
import spacy
import json
import re


def get_training_files(n_models):
    files = []
    for r, d, f in os.walk("profiles/training"):
        for profile in f:
            files.append(os.path.join(r, profile))

    return files[:n_models]


def annotate_file(profile, annotations):
    print("Annotating file {}".format(profile))
    with open(profile) as file:
        lines = file.readlines()

    annotated_data = []
    for line in lines:
        annotated_data.append(annotate_line(line, annotations))

    print("Finished annotating file {}".format(profile))
    return annotated_data


def annotate_line(line, annotations):
    nlp = spacy.blank("en")
    doc = nlp(line)

    processed_data = [line]
    lower_line = str.lower(line)
    entities = []
    for token in doc:
        text = str.lower(token.text)
        if text in annotations:
            occurs = [(m.start(), m.end()) for m in re.finditer(text, lower_line)]
            for occurrence in occurs:
                entity = [occurrence[0], occurrence[1], annotations[text]]
                entities.append(entity)
    processed_data.append({"entities": entities})
    return processed_data


def get_annotations():
    with open("data/annotations_dict.json") as f:
        annotations = json.loads(f.read())
        return annotations


def train_model(n_models):
    print("Training Model")

    if os.path.exists("data/training.json"):
        print("Deleting old training data")
        os.remove("data/training.json")

    files = get_training_files(n_models)
    annotations = get_annotations()

    annotated_data = []
    for file in files:
        annotated_data += annotate_file(file, annotations)

    with open("data/training.json", "w+") as f:
        f.write(json.dumps(annotated_data, indent=4))

    print("Training complete")