import spacy
import random
import json
import training


def train():
    with open("data/training.json") as f:
        TRAINING_DATA = json.loads(f.read())

    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
    ner.add_label("TECH")

    nlp.begin_training()

    for itn in range(10):
        random.shuffle(TRAINING_DATA)
        losses = {}

        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            texts = [text for text, entities in batch]
            annotations = [entities for text, entities in batch]

            nlp.update(texts, annotations, losses=losses)
            print(losses)

    nlp.to_disk("models/")


def run():
    test_data = "Tony likes C# and Java but hates React"

    nlp = spacy.load("models/")
    doc = nlp(test_data)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])





if __name__ == "__main__":
    files = training.train_model()
    #run()

