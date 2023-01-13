import typing as tp
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

unicode_replace = {
    8217: "'"
}


def filter_nonascii(text: str):
    for code, repl in unicode_replace.items():
        text = text.replace(chr(code), repl)
    return text


def construct_dict(dialog_id, role, utter, sentiment, history):
    return {
        'dialog_id': dialog_id,
        'role': role,
        'utterance': utter,
        'sentiment': int(sentiment) + 2,
        'history': ' EOS '.join(history)
    }


def process_file(file: Path, db: tp.List[tp.Dict[str, tp.Any]]):
    dialog_id = file.stem
    with file.open() as f:
        history = []
        for line in f:
            if line == '\n':
                break
            role, _, utter_sentiment = line[:-1].split(maxsplit=2)
            utter, sentiment = utter_sentiment.rsplit(maxsplit=1)
            try:
                int(sentiment)
            except ValueError:
                print(f'Error with sentiment mark on dialog {dialog_id}, '
                      f'file {file}, please check {line[:-1]}')
                raise
            utter = filter_nonascii(utter)
            db.append(
                construct_dict(dialog_id, role, utter, sentiment, history)
            )
            history.append(f'0.0 {utter}')


def save_db(db, out_tsv):
    db_df = pd.DataFrame(db)
    db_df['utterance'] = '1.0 ' + db_df['utterance']
    db_df = db_df[db_df['history'].str.len() > 0]
    db_df[['history', 'utterance', 'sentiment']].to_csv(out_tsv,
                                                        header=False,
                                                        index=False,
                                                        sep='\t')


def main(db_path: str, out_train_tsv: str, out_test_tsv: str,
         seed: int, test_prob: float):
    np.random.seed(seed)
    db_path = Path(db_path) / 'InteractiveSentimentDataset'
    db_train = []
    db_test = []
    total = len(list(db_path.iterdir()))
    for file in tqdm(db_path.iterdir(), total=total):
        try:
            if np.random.rand(1) < test_prob:
                process_file(file, db_test)
            else:
                process_file(file, db_train)
        except BaseException:
            print(f'Problem with file {file}')
            print('If there is a unicode error, please fix it manually')
            raise
    save_db(db_train, out_train_tsv)
    save_db(db_test, out_test_tsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=True,
                        help='Path to ScenarioDB folder')
    parser.add_argument('--output_train_tsv', default='scenario_train.tsv',
                        help='Path to store tsv database for train')
    parser.add_argument('--output_test_tsv', default='scenario_test.tsv',
                        help='Path to store tsv database for test')
    parser.add_argument('--seed', default=42,
                        help='Seed for test/train split')
    parser.add_argument('--test_part', default=0.1,
                        help='Part of all data to take for test')

    args = parser.parse_args()

    main(args.db_path, args.output_train_tsv, args.output_test_tsv,
         args.seed, args.test_part)
