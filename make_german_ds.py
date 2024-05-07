from datasets import load_dataset
from datasets import Dataset


# Dowloads datasets "bjoernp/tagesschau-2018-2023", "gnad10", and "SinclairSchneider/deutschlandfunk_de" datasets
# from Hugging Face Hub, combienes them, shuffles them and saves then as a Dataset with an 80:20 train-test-split
def make_german_ds(save_dir):
    tagesschau_ds = load_dataset("bjoernp/tagesschau-2018-2023")
    gnad10_ds = load_dataset("gnad10")
    dlf_ds = load_dataset("SinclairSchneider/deutschlandfunk_de")

    corpus = tagesschau_ds['train']['article']
    corpus.extend(gnad10_ds['train']['text'])
    corpus.extend(gnad10_ds['test']['text'])
    corpus.extend(dlf_ds['train']['content'])

    german_ds = Dataset.from_dict({"text": corpus})
    german_ds = german_ds.train_test_split(test_size=0.2, shuffle=True)

    german_ds.save_to_disk(save_dir)


def main():
    make_german_ds("german_ds")


if __name__ == '__main__':
    main()
