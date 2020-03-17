import spacy
import training
import os
import shutil


def run(n_models):
    test_files = get_working_files()
    nlp = spacy.load("models/")

    print("Preparing Output Folder")
    if os.path.exists("profiles/output"):
        shutil.rmtree("profiles/output")
    os.mkdir("profiles/output")

    for file in test_files:
        print("Processing {}".format(file))
        file_data = read_file(file)
        doc = nlp(file_data)
        write_output_file(file, n_models, doc.ents)


def write_output_file(file, n_models, ents):
    print("Writing output for: {}".format(file))

    file_name = os.path.basename(file)
    new_path = "profiles/output/{}_training_profile_{}".format(n_models, file_name)

    text = read_file(file)

    output = annotate_output(text, ents)

    with open(new_path, "w+") as f:
        f.write(output)
        f.close()


def annotate_output(text, ents):
    ents_list = [ent for ent in ents]
    ents_list.sort(key=get_end_char, reverse=True)

    for ent in ents_list:
        new_text = text[:ent.end_char] + " [TECH] " + text[ent.end_char:]
        text = new_text

    return text


def get_end_char(ent):
    return ent.end_char


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
    n_models = 1
    training.train_model(True, n_models)
    run(n_models)

