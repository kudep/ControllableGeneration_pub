import argparse
import json
from pathlib import Path


def parse_files(data_path):
    data_dir = Path(data_path)
    dialogues = []
    emotions = []
    dialog_acts = []
    topics = []
    for file in data_dir.iterdir():
        file_tags = file.name.split('_')
        if 'dialogues' not in file_tags:
            continue
        dst = dialogues
        if 'emotion' in file_tags:
            dst = emotions
        elif 'act' in file_tags:
            dst = dialog_acts
        elif 'topic' in file_tags:
            dst = topics

        with file.open('r') as f:
            for line in f:
                dst.append(line)
    return dialogues, emotions, dialog_acts, topics


def main(data_path: str, output_json: str):
    dialogues, emotions, dialog_acts, topics = parse_files(data_path)
    data = []
    for dialog_id, (dialog, emo, acts) in \
            enumerate(zip(dialogues, emotions, dialog_acts)):
        utt_texts = dialog.split('__eou__')[:-1]
        assert len(utt_texts) == len(emo.split())
        assert len(emo.split()) == len(acts.split())
        utterances = [
            {
                "utterance": utterance.strip(),
                "emotion": emotion,
                "dialog_act": dialog_act
            } for utterance, emotion, dialog_act in \
            zip(utt_texts, emo.split(), acts.split())
        ]
        data.append({
            'utterances': utterances,
        })
        if topics:
            data[-1]['topic'] = topics[dialog_id]
    with open(output_json, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='Path to directory with DailyDialog files\n'
                             '(may be path to train/test/val directory)')
    parser.add_argument('--output_json', default='daily_dialog.json',
                        help='Path and name for result json')
    args = parser.parse_args()

    main(args.data_path, args.output_json)
