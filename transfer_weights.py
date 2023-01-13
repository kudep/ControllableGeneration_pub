import argparse
import torch
from lsp_model import GPT2ConfigWithPals, GPT2LMHeadModelWithPals
from gpt2_training.train_utils import fix_state_dict_namespace

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--branch_from', type=int, required=True)
parser.add_argument('--branch_to', type=int, required=True)
parser.add_argument('--init_checkpoint', type=str, required=True)
parser.add_argument('--config_path', type=str,
                    default='models/small/config.json')
parser.add_argument('--output_model', type=str, default='transferred.pkl')

args = parser.parse_args()


def filter_weights(weights, prefix):
    filtered = {}
    for key in weights:
        if key.startswith(prefix):
            filtered[key[len(prefix):]] = weights[key]
    return filtered


config = GPT2ConfigWithPals.from_json_file(args.config_path)
model = GPT2LMHeadModelWithPals(config)
model.load_state_dict(
    fix_state_dict_namespace(torch.load(args.init_checkpoint)), strict=False)

weights = fix_state_dict_namespace(torch.load(args.weights_path))

for i, module in enumerate(model.transformer.h):
    path = f'transformer.h.{i}.branches.{args.branch_from}.'
    module.branches[args.branch_to].load_state_dict(
        filter_weights(weights, path))

torch.save(model.state_dict(), args.output_model)
