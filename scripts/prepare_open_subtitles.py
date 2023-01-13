import argparse
import json
import pandas as pd
import numpy as np
from unidecode import unidecode
import typing as tp

emotion_mapping = {
    "neutral": 1,
    "angry": 2,
    "apprehensive": 3,
    "confident": 4,
    "hopeful": 5,
    "jealous": 6,
    "annoyed": 7
}


def filter_nonascii(text: str):
    return unidecode(text)


def construct_dict(history, utterance, emotion, dialog_act):
    return {
        'history': filter_nonascii(' EOS '.join(history)),
        'utterance': filter_nonascii(utterance),
        'emotion': emotion_mapping[emotion],  # [1, 7]
        'dialog_act': int(dialog_act),  # [1, 4]
    }


def to_tsv(df, output_tsv):
    df.to_csv(
        output_tsv,
        header=False,
        index=False,
        sep='\t')


def main(db_path, output_tsv, first_meta='emotion', second_meta='dialog_act',
         train_test_split=False, test_part=0.2, seed=42):
    with open(db_path, 'r') as f:
        raw_data = json.load(f)
    db = []
    for (hist, utt), da, emo in raw_data:
        if emo not in emotion_mapping:
            continue
        db.append(construct_dict([hist], utt, emo, da))
    db_df = pd.DataFrame(db)
    db_df = db_df[db_df['history'].str.len() > 0]
    columns = ['history', 'utterance']
    if first_meta != 'none':
        columns.append(first_meta)
    if second_meta != 'none':
        columns.append(second_meta)
    if train_test_split:
        fname, ext = output_tsv.rsplit('.', maxsplit=1)
        train_out = f'{fname}_train.{ext}'
        test_out = f'{fname}_test.{ext}'

        np.random.seed(seed)
        test = np.random.choice(np.arange(len(db_df)),
                                int(test_part * len(db_df)))
        train = [i for i in range(len(db_df)) if i not in test]

        to_tsv(db_df.iloc[train][columns], train_out)
        to_tsv(db_df.iloc[test][columns], test_out)
    else:
        to_tsv(db_df[columns], output_tsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='Path to .json file')
    parser.add_argument('--output_tsv', default='open_subtitles.tsv',
                        help='Path to store tsv database for train')
    parser.add_argument('--first_meta', default='emotion',
                        choices=['emotion', 'dialog_act', 'none'],
                        help='First information column to store')
    parser.add_argument('--second_meta', default='dialog_act',
                        choices=['emotion', 'dialog_act', 'none'],
                        help='Second information column to store')
    parser.add_argument('--train_test_split', action='store_true',
                        help='Split dataset on test and train')
    parser.add_argument('--test_part', default=0.2,
                        help='Part of all dataset for test')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    main(args.data_path, args.output_tsv, args.first_meta,
         args.second_meta, args.train_test_split,
         args.test_part, args.seed)
