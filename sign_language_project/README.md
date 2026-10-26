# Sign Language Recognition (CNN from scratch)

## Person 1: dataset and preprocessing

Download the Kaggle "ASL Alphabet" dataset and place the training folders so that each class is a sub-folder:

    data/raw/asl_alphabet_train/A/*.jpg ... Z/, del/, nothing/, space/

Run everything from the project root.

    pip install -r requirements.txt
    python -m preprocessing.preprocess_images --raw_dir data/raw/asl_alphabet_train --max_per_class 500
    python -m preprocessing.dataset            # prints split statistics
    python tests/test_preprocessing.py         # runs the Person 1 tests

Team data contract: X is float32 (N, 1, 32, 32) in [0, 1], y is int64 (N,), class IDs come from data/processed/classes.json.
