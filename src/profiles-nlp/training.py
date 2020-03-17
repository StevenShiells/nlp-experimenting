import os
import spacy
import json
import re
import random
from spacy.pipeline import EntityRuler
import shutil


def get_training_files(n_models):
    files = []
    count = 0
    for r, d, f in os.walk("profiles/training"):
        for profile in f:
            count += 1
            if count > n_models:
                return files
            files.append(os.path.join(r, profile))

    return files


def annotate_file(profile, annotations):
    print("Annotating file {}".format(profile))
    with open(profile, encoding='utf8') as file:
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

    annotations.sort(key=get_annotation_length, reverse=True)

    punctuation_regex = "[\s,!?.:]{1}"
    whitespace_regex = "[\s]+"

    for annotation in annotations:
        process_regex("{}{}{}".format(whitespace_regex, re.escape(annotation), punctuation_regex), lower_line, entities, 1)
        process_regex("^{}{}".format(re.escape(annotation), punctuation_regex), lower_line, entities, 0)
    processed_data.append({"entities": entities})
    return processed_data


def process_regex(regex, lower_line, entities, start_offset):
    occurs = [(m.start(), m.end()) for m in re.finditer(regex, lower_line)]
    for occurrence in occurs:
        entity = [occurrence[0] + start_offset, occurrence[1] - 1, 'TECH']
        already_matched = False
        for ent in entities:
            if (ent[0] <= entity[0] <= ent[1]) or (ent[0] <= entity[1] <= ent[1]):
                already_matched = True
                break

        if not already_matched:
            entities.append(entity)


def get_annotation_length(annotation):
    return len(annotation)


def get_annotations():
    with open("data/tech_list") as f:
        raw_techs = f.readlines()
        annotations = []
        for raw_tech in raw_techs:
            tech = str.lower(raw_tech).strip("\n")
            if tech not in annotations:
                annotations.append(tech)

        return annotations


def get_patterns():
    with open("data/patterns.json") as f:
        patterns = json.loads(f.read())
        return patterns


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

    patterns = get_patterns()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    #nlp.add_pipe(ruler)

    ner.add_label('TECH')

    nlp.begin_training()

    for itn in range(10):
        print("Training iteration: {}".format(itn))
        random.shuffle(TRAINING_DATA)
        losses = {}

        for batch in spacy.util.minibatch(TRAINING_DATA, size=10):
            texts = [text for text, entities in batch]
            annotations = [entities for text, entities in batch]

            nlp.update(texts, annotations, losses=losses)

    print("Clearing old model")
    shutil.rmtree("models/")

    nlp.to_disk("models/")


    print("Model Training Completed")