#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
'''

import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch

import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size

from lsp_model import GPT2Tokenizer, Adam, GPT2ConfigWithPals, \
    GPT2LMHeadModelWithPals
from gpt2_training.train_utils import load_model, boolean_string, set_lr, \
    get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss, eval_model_all_outputs, \
    eval_model_loss_by_branches

from data_loader import BucketingDataLoader, DynamicBatchingLoader, \
    DistributedBucketingDataLoader, MixBucketingDataLoader

from gpt2_training.distributed import all_reduce_and_rescale_tensors, \
    all_gather_list

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str, nargs='+')
parser.add_argument("--train_input_file", type=str, nargs='+')
parser.add_argument("--eval_input_file", type=str, nargs='+')
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=10000,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True,
                    help='turn on progress bar')
parser.add_argument('--eval_only', type=boolean_string, default=False)
parser.add_argument('--fixed_branches', type=int, nargs='+', default=[])
parser.add_argument('--working_pals', type=int, nargs='+', default=[])
parser.add_argument('--eval_default_branches', type=int, nargs='+', default=[])
parser.add_argument('--eval_by_branch', action='store_true')
parser.add_argument('--gen_eval_out', action='store_true')
parser.add_argument('--eval_out_path', type=str, default='model_answer.csv')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--trainable_tasks', type=int, nargs='+', default=[])
parser.add_argument('--train_base', type=boolean_string, default=False)
parser.add_argument('--default_branch_train_prob', type=float, default=0.2)
parser.add_argument('--reverse_meta', type=boolean_string, default=False)

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')

# do normal parsing
args = parser.parse_args()

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)

assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)
logger.info('train batch size = {}, '
            'new train batch size (after gradient accumulation) = {}'.format(
    args.train_batch_size * args.gradient_accumulation_steps,
    args.train_batch_size))

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(
    args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2ConfigWithPals.from_json_file(
    join(args.model_name_or_path, 'config.json'))

if args.local_rank == -1:
    if not args.eval_only:
        if len(args.train_input_file) == 1:
            train_dataloader = BucketingDataLoader(args.train_input_file[0],
                                                   args.train_batch_size,
                                                   args.max_seq_length,
                                                   reverse_meta=args.reverse_meta)
        else:
            train_dataloader = MixBucketingDataLoader(args.train_input_file,
                                                      args.train_batch_size,
                                                      args.max_seq_length)
else:
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length,
        reverse_meta=args.reverse_meta)

if len(args.eval_input_file) > 1 and not args.eval_only:
    print('Error: multiple eval files only in eval_only mode')
    exit(0)

if not args.eval_only:
    args.eval_input_file = args.eval_input_file[0]
    args.init_checkpoint = args.init_checkpoint[0]

    eval_dataloader_loss = DynamicBatchingLoader(
        args.eval_input_file, enc, args.normalize_data,
        args.eval_batch_size, args.max_seq_length)

    eval_dataloader_gen = get_eval_list_same_length(
        args.eval_input_file, enc, args.eval_batch_size, True)

#########################################################################
# Prepare Model and Optimizer
##########################################################################
if args.eval_only:
    model = load_model(GPT2LMHeadModelWithPals(config), args.init_checkpoint[0],
                       args, verbose=True)
else:
    model = load_model(GPT2LMHeadModelWithPals(config), args.init_checkpoint,
                       args, verbose=True)
if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#if args.fp16:
if False:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex.fp16_utils import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False)
                          #max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                                   verbose=False)
    else:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   verbose=False)
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                     max_grad_norm=1.0)
    if args.fp16:
        import apex.amp as amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

# torch.autograd.set_detect_anomaly(True)
#########################################################################
# Evaluation
##########################################################################

true_model = model

if not isinstance(model, GPT2LMHeadModelWithPals):
    true_model = model.module

true_model.set_pals_work_state(args.working_pals)
true_model.setup_trainable_parts(args.trainable_tasks,
                                 train_base_model=args.train_base)

if args.eval_only:
    results = {}
    for init_checkpoint in args.init_checkpoint:
        model = load_model(GPT2LMHeadModelWithPals(config),
                           init_checkpoint,
                           args, verbose=True)
        true_model = model

        if not isinstance(model, GPT2LMHeadModelWithPals):
            true_model = model.module

        true_model.set_pals_work_state(args.working_pals)
        true_model.setup_trainable_parts(args.trainable_tasks,
                                         train_base_model=args.train_base)
        for eval_input_file in args.eval_input_file:
            eval_out_path = args.eval_out_path
            if eval_out_path == "__auto__":
                eval_parts = init_checkpoint.split('/')
                eval_out_path = eval_parts[0]
                if eval_parts[0] == 'models':
                    eval_out_path = eval_parts[1]
                num_iterations = eval_parts[-1].split('.')[-2].split('-')[-1]
                try:
                    int(num_iterations)
                except ValueError:
                    num_iterations = '0'
                eval_out_path = '_'.join([eval_out_path, num_iterations])

            if args.gen_eval_out:
                eval_dataloader_loss = DynamicBatchingLoader(
                    eval_input_file, enc, args.normalize_data,
                    1, args.max_seq_length)
            else:
                eval_dataloader_loss = DynamicBatchingLoader(
                    eval_input_file, enc, args.normalize_data,
                    args.eval_batch_size, args.max_seq_length)

            if args.fixed_branches == [-1]:
                variants = [[]]
                for branch_size in config.branch_structure:
                    new_variants = []
                    for variant in variants:
                        for i in range(branch_size):
                            new_variants.append(variant + [i])
                    variants = new_variants
            else:
                variants = [args.fixed_branches]
            for fixed_branches in variants:
                # print(f'Evaluating {fixed_branches}')
                if fixed_branches:
                    true_model.choose_workers_in_branches(fixed_branches)

                if args.gen_eval_out:
                    eval_out_path = eval_out_path + \
                                    ('_fixed_' if fixed_branches else '') + \
                                    '_'.join([str(b) for b in fixed_branches]) + \
                                    ('_default_' if args.eval_default_branches else '') + \
                                    '_'.join([str(b) for b in args.eval_default_branches]) + '.csv'
                    print(f'Evaluating to {eval_out_path}')
                    eval_model_all_outputs(model, enc, eval_dataloader_loss, 0,
                                           eval_out_path, args,
                                           change_branches=(fixed_branches == []),
                                           default_branches=args.eval_default_branches)
                elif args.eval_by_branch:
                    eval_loss, eval_ppl = eval_model_loss_by_branches(
                        model, enc, eval_dataloader_loss, 0, args,
                        change_branches=(fixed_branches == []),
                        default_branches=args.eval_default_branches)
                    for meta in eval_ppl:
                        results[f'{eval_input_file}_{meta}'] = (
                            eval_loss[meta], eval_ppl[meta])
                else:
                    eval_loss, eval_ppl = eval_model_loss(
                        model, enc, eval_dataloader_loss, 0, args,
                        change_branches=(fixed_branches == []),
                        default_branches=args.eval_default_branches)
                    results[f'{eval_input_file}_{fixed_branches}_{init_checkpoint}'] = (
                        eval_loss, eval_ppl)
    for name, (loss, ppl) in results.items():
        print(f'{name}: loss: {loss}    ppl: {ppl}')
    exit(0)

#########################################################################
# Training !
##########################################################################

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
    eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
    print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,eval_loss,eval_ppl', file=eval_logger)

global_step = 0
step = 0
epoch = 0

if args.continue_from:
    global_step = args.continue_from
    step = global_step * 2 - 1

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps - global_step, desc=f"training")
    else:
        pbar = None

while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples,
     nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        if len(args.train_input_file) > 1:
            dataset_id, batch = batch
            true_model.setup_trainable_parts([dataset_id],
                                             train_base_model=args.train_base)
        # activate new training mode
        seq_len = batch[0].shape[1]
        # batch = tuple(t.to(device) for t in batch)
        input_ids, position_ids, token_ids, label_ids, meta, *_ = batch
        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        token_ids = token_ids.to(device)
        label_ids = label_ids.to(device)
        if args.no_token_id:
            token_ids = None
        # meta[0] because all metas should be the same for one batch
        branches = [int(br) for br in meta[0]]
        # training default branch
        for i in range(len(branches)):
            if np.random.choice([False, True],
                                p=[1 - args.default_branch_train_prob,
                                   args.default_branch_train_prob]):
                branches[i] = 0  # default branch idx
        true_model.choose_workers_in_branches(branches)
        loss, ppl = model(input_ids, position_ids, token_ids, label_ids)

        if n_gpu > 1:
            loss = loss.mean()
            ppl = ppl.mean()
        loss = loss / (args.train_batch_size / input_ids.shape[0])
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            # optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += float(loss.item()) * (
                args.train_batch_size / input_ids.shape[0])
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps
        if ppl.item() < INF:
            tr_ppl += ppl.item()
        else:
            tr_ppl += mean_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        # gradient update
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Print log info to file
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"tok/s: {n_token_real_all_proc // epoch_time // 1000}k "
                        f"ppl: {mean_ppl:.2f} epoch: {epoch}")
                    pbar.update(1)
                print('{},{},{},{},{},{},{},{}'.format(
                    epoch + 1, global_step + 1, step + 1, mean_loss, mean_ppl,
                    n_token_real_all_proc, n_token_total_all_proc, epoch_time),
                    file=train_logger)

            if global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    torch.save(
                        {k: (v.cpu() if v is not None else None)
                         # save to cpu tensors
                         for k, v in model.state_dict(keep_vars=True).items()},
                        join(output_dir,
                             f'GP2-pretrain-step-{global_step}.pkl'))

                    eval_loss, eval_ppl = eval_model_loss(
                        model, enc, eval_dataloader_loss, epoch, args,
                        change_branches=True)
                    # enable generation step evaluation for now
                    # gen_response = eval_model_generation(
                    #     model, enc, eval_dataloader_gen, epoch, args)
                    '''
                    # probably use beam search only for test set
                    if False:
                        gen_response_beam = eval_model_generation(
                            model, enc, eval_dataloader_gen, epoch, args,
                            use_beam_search=True, beam_width=3)
                    '''
                    print('{},{},{},{},{}'.format(
                        epoch + 1, global_step + 1, step + 1, eval_loss,
                        eval_ppl),
                        file=eval_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()
            if global_step >= args.num_optim_steps:
                break

        if (step + 1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if global_step >= args.num_optim_steps:
        break
    epoch += 1

if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()
