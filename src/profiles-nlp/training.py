import os
import spacy
import json
import re
import random


def get_training_files(n_models):
    files = []
    count = 0
    for r, d, f in os.walk("profiles/training"):
        for profile in f:
            files.append(os.path.join(r, profile))
            count += 1
            if count > n_models:
                return files

    return files


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
    processed_data = [line]
    lower_line = str.lower(line)
    entities = []
    for annotation in annotations:
        occurs = [(m.start(), m.end()) for m in re.finditer(annotation, lower_line)]
        for occurrence in occurs:
            entity = [occurrence[0], occurrence[1], annotations[annotation]]
            entities.append(entity)
    processed_data.append({"entities": entities})
    return processed_data


def get_annotations():
    with open("data/annotations_dict.json") as f:
        annotations = json.loads(f.read())
        return annotations


def generate_training_data(n_models):
    print("Generate Training Data")

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

    print("Training Data Generated")


def train_model(generate_data=False, n_models=100):
    if generate_data:
        generate_training_data(n_models)

    print("Training Model")
    with open("data/training.json") as f:
        TRAINING_DATA = json.loads(f.read())

    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)

    annotations = get_annotations()
    annotation_labels = set()
    for annotation in annotations:
        annotation_labels.add(annotations[annotation])

    for annotation_label in annotation_labels:
        ner.add_label(annotation_label)

    nlp.begin_training()

    for itn in range(10):
        print("Training iteration: {}".format(itn))
        random.shuffle(TRAINING_DATA)
        losses = {}

        for batch in spacy.util.minibatch(TRAINING_DATA, size=10):
            texts = [text for text, entities in batch]
            annotations = [entities for text, entities in batch]

            nlp.update(texts, annotations, losses=losses)

    nlp.to_disk("models/")
    print("Model Training Completed")