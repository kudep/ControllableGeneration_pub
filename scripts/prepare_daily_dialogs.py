import argparse
import json
import pandas as pd

from nltk.tokenize import sent_tokenize

import typing as tp

unicode_replace = {
    8217: "'",
    8212: "-",
    8220: "\"",
    8221: "\"",
    12290: "",
    8216: "'",
    8211: "-",
    8242: "'",
    12289: "",
}

sf_mapping = {}


def filter_nonascii(text: str):
    for code, repl in unicode_replace.items():
        text = text.replace(chr(code), repl)
    return text


def construct_dict_sf(dialog_id, utterance, emotion, speech_function, history):
    if speech_function not in sf_mapping:
        sf_mapping[speech_function] = len(sf_mapping)
    return {
        'dialog_id': dialog_id,
        'utterance': '1.0 ' + filter_nonascii(utterance),
        'emotion': emotion,
        'speech_function': sf_mapping[speech_function],
        'history': filter_nonascii(' EOS '.join(history))
    }


def construct_dict_da(dialog_id, utterance, emotion, dialog_act,
                      history, no_weights=False, **_):
    return {
        'dialog_id': dialog_id,
        'utterance': ('' if no_weights else '1.0 ') +
                     filter_nonascii(utterance),
        'emotion': int(emotion) + 1,  # [0, 6] + 1
        'dialog_act': int(dialog_act),  # [1, 4]
        'history': filter_nonascii(' EOS '.join(history))
    }


def process_dialog_sf(dialog, dialog_id: int,
                      db: tp.List[tp.Dict[str, tp.Any]]):
    history = []
    for utterance in dialog['utterances']:
        final_answer = utterance['utterance']
        sentences = sent_tokenize(final_answer)
        if len(sentences) != len(utterance['speech_functions']):
            print(f'Trouble with dialog {dialog_id}:')
            print(f'Sentence tokenization: {sentences}')
            print(f'Speech functions: {utterance["speech_functions"]}')
            continue
        history.append('')  # For final EOS to appear at the end of tgt
        for sent, sf in zip(sentences, utterance["speech_functions"]):
            db.append(
                construct_dict_sf(dialog_id=dialog_id,
                                  history=history,
                                  emotion=utterance['emotion'],
                                  speech_function=sf,
                                  utterance=sent)
            )
            if not history[-1]:
                history[-1] = '0.0'
            history[-1] = ' '.join([history[-1], sent])


def process_dialog_da(dialog, dialog_id: int,
                      db: tp.List[tp.Dict[str, tp.Any]],
                      no_weights=False):
    history = []
    for utterance in dialog['utterances']:
        db.append(
            construct_dict_da(dialog_id=dialog_id, history=history,
                              no_weights=no_weights, **utterance)
        )
        if no_weights:
            history.append(utterance["utterance"])
        else:
            history.append(f'0.0 {utterance["utterance"]}')


def main(data_path: str, output_tsv: str, mapping_path: str,
         first_meta='emotion', second_meta='speech_function',
         no_weights=False):
    if second_meta not in ['speech_function', 'dialog_act', 'none']:
        print(f'Second meta "{second_meta}" is unknown')
        return

    with open(data_path) as f:
        data = json.load(f)
    db = []
    for dialog_id, dialog in enumerate(data):
        if second_meta == 'speech_function':
            process_dialog_sf(dialog, dialog_id, db)
            with open(mapping_path, 'w') as f:
                json.dump({value: key for key, value in sf_mapping.items()}, f)
        elif second_meta in ['dialog_act', 'none']:
            process_dialog_da(dialog, dialog_id, db, no_weights=no_weights)
    db_df = pd.DataFrame(db)
    db_df = db_df[db_df['history'].str.len() > 0]
    columns = ['history', 'utterance']
    if first_meta != 'none':
        columns.append(first_meta)
    if second_meta != 'none':
        columns.append(second_meta)
    db_df[columns].to_csv(
        output_tsv,
        header=False,
        index=False,
        sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='Path to .json file')
    parser.add_argument('--output_tsv', default='test_daily_dialog.tsv',
                        help='Path to store tsv database for train')
    parser.add_argument('--mapping_path', default='mapping.json',
                        help='Path to store mapping for speech functions')
    parser.add_argument('--first_meta', default='emotion',
                        choices=['emotion', 'dialog_act', 'none'],
                        help='First information column to store')
    parser.add_argument('--second_meta', default='speech_function',
                        choices=['speech_function', 'dialog_act', 'none'],
                        help='Second information column to store')
    parser.add_argument('--no_weights', action='store_true',
                        help='Add weights to utterances '
                             '(0 for history, 1 for current)')
    args = parser.parse_args()

    main(args.data_path, args.output_tsv, args.mapping_path, args.first_meta,
         args.second_meta, no_weights=args.no_weights)
