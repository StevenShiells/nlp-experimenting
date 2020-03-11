import spacy
import training
import os


def run():
    test_files = get_working_files()
    nlp = spacy.load("models/")
    for file in test_files:
        file_data = read_file(file)
        doc = nlp(file_data)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


def get_working_files():
    files = []
    for r, d, f in os.walk("profiles/untrained"):
        for profile in f:
            files.append(os.path.join(r, profile))

    return files


def read_file(file):
    print("Reading file {}".format(file))
    with open(file) as f:
        lines = f.readlines()
        return "\n".join(lines)


if __name__ == "__main__":
    training.train_model(True)
    run()

