#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import torch.nn.functional as F
import logging

import numpy as np
import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict

from tqdm import tqdm

logger = logging.getLogger(__name__)

EOS_ID = 50256


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0] / len(generated)
    BLEUscore[1] = BLEUscore[1] / len(generated)
    BLEUscore[2] = BLEUscore[2] / len(generated)
    BLEUscore[3] = BLEUscore[3] / len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g) - n):
                ngram = ' '.join(g[idx:idx + n + 1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v + 0.0) / total * (
                    np.log(v + 0.0) - np.log(total))
        div_score[n] = (len(counter[n].values()) + 0.0) / total
    return etp_score, div_score


def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args,
                    change_branches=False, default_branches=None):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    t_model = model.module if isinstance(model,
                                         torch.nn.DataParallel) else model
    if default_branches is None:
        default_branches = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(
                t if isinstance(t, list) else t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _, meta = batch
            if change_branches:
                branches = [int(br) if i not in default_branches else 0
                            for i, br in enumerate(meta[0])]
                t_model.choose_workers_in_branches(branches)
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(
        f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(
        tot_sample)


def eval_model_loss_by_branches(model, tokenizer, eval_dataloader, epoch_id,
                               args, change_branches=False, default_branches=None):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = defaultdict(list)
    tot_ppl = defaultdict(list)
    tot_sample = defaultdict(list)
    t_model = model.module if isinstance(model,
                                         torch.nn.DataParallel) else model
    if default_branches is None:
        default_branches = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(
                t if isinstance(t, list) else t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _, meta = batch
            if change_branches:
                branches = [int(br) if i not in default_branches else 0
                            for i, br in enumerate(meta[0])]
                t_model.choose_workers_in_branches(branches)
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss[tuple(meta[0])].append(loss.mean().item() * n_sample)
            tot_ppl[tuple(meta[0])].append(ppl.mean().item() * n_sample)
            tot_sample[tuple(meta[0])].append(n_sample)
    loss = {}
    ppl = {}
    for meta in tot_ppl:
        ppl[meta] = np.sum(tot_ppl[meta]) / np.sum(tot_sample[meta])
        loss[meta] = np.sum(tot_loss[meta]) / np.sum(tot_sample[meta])
        print(
            f"\n Meta {meta}: Val loss {loss[meta]} Val ppl {ppl[meta]} ")
    return loss, ppl


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values,
                             torch.ones_like(logits, dtype=logits.dtype) *
                             -float('Inf'),
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[...,
                                            :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove,
                                                   filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices,
                                                  sorted_logits)

    return logits


def eval_model_all_outputs(model, tokenizer, eval_dataloader, epoch_id,
                           eval_out_path, args, change_branches=False,
                           default_branches=None):
    if default_branches is None:
        default_branches = []
    eos = [tokenizer.encoder["<|endoftext|>"]]

    logger.info('computing model outputs on eval data, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    t_model = model.module if isinstance(model,
                                         torch.nn.DataParallel) else model
    new_data = []
    with torch.no_grad():
        top_k = -1
        top_p = 0.9
        for step, batch in tqdm(enumerate(eval_dataloader),
                                total=len(eval_dataloader)):
            batch = tuple(
                t if isinstance(t, list) else t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _, meta = batch
            if change_branches:
                branches = [int(br) if i not in default_branches else 0
                            for i, br in enumerate(meta[0])]
                t_model.choose_workers_in_branches(branches)

            history = input_ids[:, :src_len[0] + 1]
            sent = []
            for _ in range(500):
                logits, past = t_model(history)
                logits = logits[:, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)

                probs = torch.softmax(logits, dim=-1)

                prev_input = torch.multinomial(probs, num_samples=1)
                history = torch.cat((history, prev_input), dim=1)
                prev_word = prev_input.item()

                if prev_word == eos[0]:
                    break
                sent.append(prev_word)

            answer = tokenizer.decode(sent)
            new_data.append({
                'history': tokenizer.decode(
                    input_ids[0, :src_len[0]].cpu().numpy()),
                'gold_answer': tokenizer.decode(
                    input_ids[0, src_len[0] + 1:].cpu().numpy()),
                'model_answer': answer,
                'meta': meta[0]
            })
    df = pd.DataFrame(new_data)
    df.to_csv(eval_out_path)
