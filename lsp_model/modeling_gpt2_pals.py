import logging
import copy
import math
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional
import numpy as np

from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, \
    GPT2Model, GPT2LMHead, Attention, Block, \
    LayerNorm, MLP, GPT2Config
from lsp_model.modeling_gpt2 import AttentionFP16, GPT2ModelFP16, \
    GPT2LMHeadModel

logger = logging.getLogger(__name__)

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype: torch.dtype) -> float:
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


class GPT2ConfigWithPals(GPT2Config):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            branch_structure=None,
            extra_dim=None,
            n_embd_aug=204,
            pals=True,
            pals_type='base',
            blend_layer='mean',
            branch_inside='different_branches',
            work_with_pals=True,
            use_branch_classification_loss=False,
            branch_classification_lambda=0.2,
            custom_weights=None,
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        super(GPT2ConfigWithPals, self).__init__(
            vocab_size_or_config_json_file, n_positions, n_ctx, n_embd,
            n_layer, n_head, layer_norm_epsilon, initializer_range)
        self.branch_structure = branch_structure if branch_structure is not None else []
        self.blend_layer = blend_layer
        self.branch_inside = branch_inside
        self.extra_dim = extra_dim
        self.n_embd_aug = n_embd_aug
        self.pals = pals
        self.pals_type = pals_type
        self.work_with_pals = work_with_pals
        self.use_branch_classification_loss = use_branch_classification_loss
        self.branch_classification_lambda = branch_classification_lambda
        self.custom_weights = custom_weights

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GPT2Pals(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, scale=False):
        super(GPT2Pals, self).__init__()
        # Encoder and decoder matrices project down to the smaller dimension
        self.aug_dense = nn.Linear(config.n_embd, config.n_embd_aug)
        self.aug_dense2 = nn.Linear(config.n_embd_aug, config.n_embd)
        # Attention without the final matrix multiply.
        self.attn = AttentionFP16(config.n_embd_aug, config.n_ctx, config,
                                  scale)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, layer_past=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug, _ = self.attn(hidden_states_aug, layer_past)
        hidden_states = self.aug_dense2(hidden_states_aug)
        hidden_states = self.hidden_act_fn(hidden_states)
        return hidden_states


class GPT2PalsLite(nn.Module):
    def __init__(self, inp_dim=512, hid_dim=200, out_dim=512, dropout=0.1):
        super(GPT2PalsLite, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.q_lin = nn.Linear(inp_dim, hid_dim)
        self.k_lin = nn.Linear(inp_dim, hid_dim)
        self.v_lin = nn.Linear(inp_dim, hid_dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(hid_dim, out_dim)
        nn.init.xavier_normal_(self.out_lin.weight)
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, query_len, dim = hidden_states.size()
        q = self.q_lin(hidden_states)
        k = self.k_lin(hidden_states)
        v = self.v_lin(hidden_states)
        scale = math.sqrt(self.hid_dim)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)
            dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = torch.softmax(dot_prod, dim=-1,
                                     dtype=torch.float).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        attentioned = attn_weights.bmm(v)
        out = self.out_lin(attentioned)
        return out


class GPT2LowRank(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, scale=False):
        super(GPT2LowRank, self).__init__()
        # Encoder and decoder matrices project down to the smaller dimension
        if config.extra_dim:
            self.aug_dense = nn.Linear(config.n_embd, config.extra_dim)
            self.aug_dense2 = nn.Linear(config.extra_dim, config.n_embd)
        else:
            self.aug_dense = nn.Linear(config.n_embd,
                                       config.n_embd_aug)
            self.aug_dense2 = nn.Linear(config.n_embd_aug,
                                        config.n_embd)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, layer_past=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
        hidden_states = self.aug_dense2(hidden_states_aug)
        return hidden_states


class BlendLayerMean(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x.mean(dim=0)


class BlendLayerWeightedMean(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, branch_weights=None):
        super().__init__()
        self.config = config
        self.branch_weights = branch_weights
        if branch_weights is None:
            num_branches = len(self.config.branch_structure)
            self.branch_weights = [float(num_branches)] +\
                                  [1.] * num_branches
        self.branch_weights = torch.tensor(self.branch_weights).float()
        self.branch_weights /= torch.sum(self.branch_weights)

    def forward(self, x):
        branch_weights = self.branch_weights.to(x.device)
        x = x * branch_weights[:, None, None, None]
        return x.sum(dim=0)


class BlendLayerDoubleLinear(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals):
        super().__init__()
        num_branches = len(config.branch_structure)
        self.aug_dense = nn.Linear(config.n_embd * (num_branches + 1),
                                   config.n_embd_aug)
        self.aug_dense2 = nn.Linear(config.n_embd_aug,
                                    config.n_embd)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, x):
        merged = torch.cat(list(x), dim=-1)
        hidden_states_aug = self.aug_dense(merged)
        hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
        hidden_states = self.aug_dense2(hidden_states_aug)
        return hidden_states


class BlendLayerLinear(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals):
        super().__init__()
        num_branches = len(config.branch_structure)
        self.aug_dense = nn.Linear(config.n_embd * (num_branches + 1),
                                   config.n_embd)
        self.config = config

    def forward(self, x):
        merged = torch.cat(list(x), dim=-1)
        hidden_states = self.aug_dense(merged)
        return hidden_states


class BlendLayerLinearPals(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals):
        super().__init__()
        num_branches = len(config.branch_structure)
        self.dense = nn.Linear(config.n_embd * num_branches,
                                   config.n_embd)
        self.config = config

    def forward(self, x):
        original, merged_pals = x[0], torch.cat(list(x[1:]), dim=-1)
        emb = self.dense(merged_pals)
        return (original + emb) / 2


class BlendLayerDoubleLinearPals(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals):
        super().__init__()
        num_branches = len(config.branch_structure)
        self.dense = nn.Linear(config.n_embd * num_branches,
                               config.n_embd * (num_branches + 1) // 2)
        self.hidden_act_fn = gelu
        self.dense2 = nn.Linear(config.n_embd * (num_branches + 1) // 2,
                                config.n_embd)
        self.config = config

    def forward(self, x):
        original, merged_pals = x[0], torch.cat(list(x[1:]), dim=-1)
        emb = self.dense(merged_pals)
        emb = self.hidden_act_fn(emb)
        emb = self.dense2(emb)
        return (original + emb) / 2


class BranchClassifier(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, branch_population):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.n_embd, branch_population)
        self.dropout = nn.Dropout(0.2)

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, emb, label):
        emb = self.dropout(emb)
        logits = self.classifier(torch.mean(emb, dim=1))
        labels = torch.tensor(
            [label for _ in range(logits.shape[0])],
            device=emb.device)
        loss = self.loss(logits, labels)
        return logits, loss


class BranchPack(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, branch_population):
        super().__init__()
        if config.pals:
            if config.pals_type == "base":
                multi = GPT2Pals(config)
            else:
                multi = GPT2PalsLite(config.n_embd, config.n_embd_aug, config.n_embd)
        else:
            multi = GPT2LowRank(config)
        self.workers = nn.ModuleList(
                [copy.deepcopy(multi) for _ in
                 range(branch_population + 1)]
            )
        self.worker = 0

    def forward(self, *args, **kwargs):
        return self.workers[self.worker](*args, **kwargs)

    def choose_worker(self, worker):
        self.worker = worker


class OneBranchWithEmb(nn.Module):
    def __init__(self, config: GPT2ConfigWithPals, branch_population):
        super().__init__()
        if config.pals:
            if config.pals_type == "base":
                multi = GPT2Pals(config)
            else:
                multi = GPT2PalsLite(config.n_embd, config.n_embd_aug, config.n_embd)
        else:
            multi = GPT2LowRank(config)
        self.backbone = multi

        self.embeddings = nn.Embedding(branch_population + 1, config.n_embd)
        self.worker = 0

    def forward(self, x, *args, **kwargs):
        embedding = self.embeddings(torch.tensor(self.worker,
                                                 device=x.device))
        x = x + embedding
        base = self.backbone(x, *args, **kwargs)
        return base

    def choose_worker(self, worker):
        self.worker = worker


class BlockWithPals(Block):
    def __init__(self, n_ctx, config: GPT2ConfigWithPals, scale=False):
        super(BlockWithPals, self).__init__(n_ctx, config, scale)
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = AttentionFP16(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

        branch = BranchPack
        if config.branch_inside == 'different_branches':
            branch = BranchPack
        elif config.branch_inside == 'one_branch_with_emb':
            branch = OneBranchWithEmb

        self.branches = nn.ModuleList([
            branch(config, branch_population)
            for branch_population in config.branch_structure
        ])
        self.branch_classifier = nn.ModuleList([
            BranchClassifier(config, branch_population)
            for branch_population in config.branch_structure
        ])
        if config.blend_layer == 'mean':
            self.blend_layer = BlendLayerMean(config)
        elif config.blend_layer == 'linear':
            self.blend_layer = BlendLayerLinear(config)
        elif config.blend_layer == 'double_linear':
            self.blend_layer = BlendLayerDoubleLinear(config)
        elif config.blend_layer == 'weighted_mean':
            self.blend_layer = BlendLayerWeightedMean(config)
        elif config.blend_layer == 'custom_weights':
            self.blend_layer = BlendLayerWeightedMean(config, config.custom_weights)
        elif config.blend_layer == 'linear_pals':
            self.blend_layer = BlendLayerLinearPals(config)
        elif config.blend_layer == 'double_linear_pals':
            self.blend_layer = BlendLayerDoubleLinearPals(config)
        self.pals_work_state = config.work_with_pals
        self.workers_in_branches = [0] * len(self.branches)
        self.use_branch_classification_loss = config.use_branch_classification_loss

    def forward(self, x, layer_past=None):
        x_norm = self.ln_1(x.clone())
        a, present = self.attn(x_norm.clone(), layer_past=layer_past)
        losses = []
        if self.pals_work_state:
            embeddings = [a] + [branch(x_norm.clone(), layer_past)
                                for branch in self.branches]

            if self.use_branch_classification_loss:
                for i, emb in enumerate(embeddings[1:]):
                    _, loss = self.branch_classifier[i](emb, self.workers_in_branches[i])
                    losses.append(loss)
                losses = torch.tensor(losses, device=x.device)

            emb = self.blend_layer(torch.stack(embeddings))
            x += emb
        else:
            x += a
        m = self.mlp(self.ln_2(x.clone()))
        x = x + m
        return x, present, losses

    def choose_trainable_branches(self, branches=None):
        if branches is None:
            branches = []
        for i, branch in enumerate(self.branches.children()):
            branch_status = i in branches
            for param in branch.parameters():
                param.requires_grad = branch_status
        train_blender = len(branches) > 0
        for param in self.blend_layer.parameters():
            param.requires_grad = train_blender

    def set_pals_work_state(self, state=True):
        self.pals_work_state = state

    def choose_workers_in_branches(self, workers_in_branches=None):
        if workers_in_branches is None:
            workers_in_branches = [0] * len(self.branches)
        self.workers_in_branches = workers_in_branches
        for worker, pack in zip(self.workers_in_branches, self.branches):
            pack.choose_worker(worker)


class GPT2ModelFP16WithPals(GPT2ModelFP16):
    def __init__(self, config):
        super(GPT2ModelFP16WithPals, self).__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = BlockWithPals(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None,
                past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length,
                                        input_ids.size(-1) + past_length,
                                        dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        b_losses = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present, b_loss = block(hidden_states, layer_past)
            b_losses.append(b_loss)
            presents.append(present)
        if self.config.use_branch_classification_loss:
            b_losses = torch.stack(b_losses)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents, b_losses

    def choose_trainable_branches(self, branches=None):
        for block in self.h.children():
            block.choose_trainable_branches(branches)

    def set_pals_work_state(self, working_pals: Optional[List[int]] = None):
        if working_pals is None:
            working_pals = []
        for i, block in enumerate(self.h.children()):
            block.set_pals_work_state(i in working_pals)

    def choose_workers_in_branches(self, workers_in_branches=None):
        for i, block in enumerate(self.h.children()):
            block.choose_workers_in_branches(workers_in_branches)


class GPT2LMHeadModelWithPals(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2LMHeadModelWithPals, self).__init__(config)
        self.transformer = GPT2ModelFP16WithPals(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None,
                lm_labels=None, past=None):
        hidden_states, presents, b_losses = self.transformer(input_ids,
                                                             position_ids,
                                                             token_type_ids,
                                                             past)
        # import pdb; pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
                              lm_labels.view(-1))
            loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
            label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
            loss = torch.sum(loss1) / torch.sum(label_size)
            ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float()
                                       / label_size.float()))
            # ppl = torch.mean(torch.exp(torch.sum(loss1, dim=1)/label_size))
            if self.config.use_branch_classification_loss:
                b_loss = torch.mean(b_losses)
                loss = loss + self.config.branch_classification_lambda * b_loss

            return loss, ppl
        return lm_logits, presents

    def forward_pointwise(self, input_ids, position_ids=None,
                          token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents, b_losses = self.transformer(input_ids,
                                                             position_ids,
                                                             token_type_ids,
                                                             past)
        # import pdb; pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
                              lm_labels.view(-1))
            loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
            label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
            loss1 = torch.sum(loss1, dim=1) / label_size
            ppl1 = torch.exp(loss1)

            if self.config.use_branch_classification_loss:
                b_loss = torch.mean(b_losses)
                loss1 = loss1 + self.config.branch_classification_lambda * b_loss

            return loss1, ppl1
        return lm_logits, presents

    def setup_trainable_parts(self,
                              branches: Optional[List[int]] = None,
                              train_base_model=False):
        for param in self.parameters():
            param.requires_grad = train_base_model
        self.transformer.choose_trainable_branches(branches)

    def set_pals_work_state(self, working_pals: Optional[List[int]] = None):
        self.transformer.set_pals_work_state(working_pals)

    def choose_workers_in_branches(self, workers_in_branches=None):
        self.transformer.choose_workers_in_branches(workers_in_branches)
