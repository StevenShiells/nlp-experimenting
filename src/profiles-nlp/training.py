import os
import spacy
import json

def get_training_files():
    files = []
    for r, d, f in os.walk("profiles/training"):
        for profile in f:
            files.append(os.path.join(r, profile))

    return files


def annotate_file(profile, annotations):
    print("Annotating file {}".format(profile))
    lines = []
    with open(profile) as file:
        lines = file.readlines()

    for line in lines:
        annotate_line(line, annotations)

    print("Finished annotating file {}".format(profile))


def annotate_line(line, annotations):
    nlp = spacy.blank("en")
    doc = nlp(line)
    for token in doc:
        text = str.lower(token.text)
        if text in annotations:
            print("{} {}".format(text, annotations[text]))


def get_annotations():
    with open("data/annotations_dict.json") as f:
        annotations = json.loads(f.read())
        return annotations


def train_model():
    print("Training Model")

    files = get_training_files()
    annotations = get_annotations()
    for file in files:
        annotate_file(file, annotations)

    print("Training complete")