import pandas as pd
from tqdm import tqdm
import torch
from transformers import BartForSequenceClassification, BartTokenizer
import argparse
import traceback

topics = ["Phatic",
          "Other",
          "Movies_TV",
          "Music",
          "SciTech",
          "Literature",
          "Travel_Geo",
          "Celebrities",
          "Games",
          "Pets_Animals",
          "Sports",
          "Psychology",
          "Religion",
          "Weather_Time",
          "Food_Drink",
          "Politics",
          "Sex_Profanity",
          "Art_Event",
          "Math",
          "News",
          "Entertainment",
          "Fashion"]

topic_sentences = {topic: f"We are talking about {' or '.join(topic.lower().split('_'))}."
                   for topic in topics if topic != "Other"}
topic_sentences["Phatic"] = "We are talking phatic."
topic_sentences["SciTech"] = "We are talking about science or technologies."
topic_sentences_df = pd.DataFrame({'hypothesis': topic_sentences.values(), 'topic': topic_sentences.keys()})
no_other_topics = [t for t in topics if t != "Other"]

# load model pretrained on MNLI
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)



def extract_last_from_history(row, num_last=5):
    return ' '.join(row['history'].split(' EOS ')[-num_last:])


def analyse_dialog(row):
    premise = extract_last_from_history(row, num_last=5)
    input_ids = tokenizer.batch_encode_plus(
        [(premise, hypothesis) for hypothesis in topic_sentences.values()],
        padding=True, return_tensors='pt')['input_ids'].to(device)

    with torch.no_grad():
        logits = model(input_ids)[0]
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        # print(probs)
        true_prob = probs[:,1].cpu().numpy()
    return true_prob


def analyse_topics(dialogs_df):
    for i in tqdm(range(len(dialogs_df))):
        res = analyse_dialog(dialogs_df.iloc[i])
        dialogs_df.loc[i, no_other_topics] = res
        dialogs_df.loc[i, 'Topic'] = no_other_topics[res.argmax()]


def process(dataset_name):
    print(device)
    model.eval()
    df = pd.read_csv(f'{dataset_name}_df.csv')
    analyse_topics(df)
    df.to_csv(f'{dataset_name}_topics_df.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse topics for datasets')
    parser.add_argument('datasets', metavar='dataset', type=str, nargs='+',
                        help='datasets to analyse topics')
    args = parser.parse_args()

    for dataset in args.datasets:
        try:
            print(f'Processing {dataset}')
            process(dataset)
        except BaseException as e:
            print(f'Error while processing {dataset}')
            print(e)
            traceback.print_exc()