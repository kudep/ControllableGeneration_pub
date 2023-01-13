# DialoGPT with controllable attributes

This repo contains modifications for DialoGPT to add control of different attributes.

### Training

#### Preparations

To train model base DialoGPT is needed. Refer to [DialoGPT readme](README_DialoGPT.md) 
for further details.

Dataset also should be prepared according [DialoGPT readme](README_DialoGPT.md) running:
```
python3 prepro.py --corpus path/to/corpus.tsv
```
Note, that `prepro.py` in this repo is modified to handle attribute labels
to train. Dataset should be in a `.tsv` format and contain first column
with history (utterance, separated with " EOS ", with spaces), second column with target utterance, and the rest columns are for
attribute labels. Attribute label should an integer from 1 to N, where N is a 
number of classes for this attribute. File shouldn't contain column names and indexes. 
Example row for two attributes is:
```
Hi , good morning , Miss ? what can I help you with ? EOS Good morning I'd like to mail this box of books to Taiwan .	OK , please put it on this scale.Airmail or by sea ?	2	2
```
History contains two utterances, and for each attribute gold label is `2`. 
The same format is needed for validation part, but no need to run `prepro.py`
this time.

#### Train

To train model `LSP_train.py` could be used in the same way as for DialoGPT, but with more arguments.
For example take a look at `scripts/train.sh`. This script launch a train for model with control
of sentiment and dialog acts using average blending for PALs.

For different blending strategies and other settings see example configs in `configs/`.

#### Transfer

If PALs weights transfer is needed, take a look at `transfer_weights.py`

### Evaluation
To evaluate model take a look at `scripts/eval.sh`. If only perplexity is needed, 
then comment out last two options. Otherwise, script will generate model responses
for all validation dataset to further classification and evaluation of control abilities.

To evaluate control abilities use `scripts/analyse.py`, requirements are in `requirements_analyse.txt`.
Our evaluation if for dialog act and sentiment attributes, download 
[dialog act classifier](https://drive.google.com/file/d/1cQOXKzNyiUKlwezYiISvn008a6R1GIpe/view?usp=sharing) to `MODEL_PATH` from `scripts/dialog_act_no_hist.json` and
[sentiment classifier](https://drive.google.com/file/d/13igJJVW1JoqOgW-zB3ABxKv_AqWvEY7V/view?usp=sharing) to `MODEL_PATH` from `scripts/sentiment_hist.json`.
Script will generate folder with all results including balanced accuracy scores and confusion matrices.

### Pretrained models

Some of our models:
- DialoGPT-small, dialog act control, average blending: [weights](https://drive.google.com/file/d/1ko8uHKOrs9-MGtFPMq7KfKkc1__m73fz/view?usp=sharing)
- DialoGPT-small, sentiment control, average blending: [weights](https://drive.google.com/file/d/1Gzjmnr2yLjkXJ17tTUzH70sbAAYHRmIa/view?usp=sharing)
- DialoGPT-small, sentiment and dialog act control, average blending: [weights](https://drive.google.com/file/d/1YQ14cJek7_37HPAEUa2xLtz6cOvZMinD/view?usp=sharing)
- DialoGPT-small, sentiment and dialog act control, weighted average blending (after transfer): [weights](https://drive.google.com/file/d/1tXfkczvYjmA23r4YNfPp9scPXN3arBlZ/view?usp=sharing)
- DialoGPT-small, sentiment and dialog act control, combination of dense and average blending (after transfer): [weights](https://drive.google.com/file/d/1xiUerxUx_B_YAd5NJT4MrTSr29ub8lzp/view?usp=sharing)
