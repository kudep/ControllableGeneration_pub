# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import torch
from typing import Tuple, List, Optional, Union, Dict, Set

import numpy as np
from transformers import AutoTokenizer, BertTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('dialog_act_hist_preprocessor')
class DialogActHistPreprocessor(Component):

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        special_tokens_dict = {'additional_special_tokens': ['<utt>', '</utt>', '<begin>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

    def __call__(self, texts_a: List[List[str]], texts_b: List[List[str]] = None) -> Dict[str, torch.tensor]:

        lengths = []
        for text_a, text_b in zip(texts_a, texts_b):
            text_b = f"<utt> {text_b} </utt>"
            encoding = self.tokenizer.encode_plus(text=text_a, text_pair=text_b,
                                                  return_attention_mask=True, add_special_tokens=True,
                                                  truncation=True)
            lengths.append(len(encoding["input_ids"]))
        max_len = max(lengths)
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        for text_a, text_b in zip(texts_a, texts_b):
            text_b = f"<utt> {text_b} </utt>"
            encoding = self.tokenizer.encode_plus(text=text_a, text_pair=text_b,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            input_ids_batch.append(encoding["input_ids"])
            attention_mask_batch.append(encoding["attention_mask"])
            token_type_ids_batch.append(encoding["token_type_ids"])
            
        input_features = {"input_ids": torch.LongTensor(input_ids_batch),
                          "attention_mask": torch.LongTensor(attention_mask_batch),
                          "token_type_ids": torch.LongTensor(token_type_ids_batch)}
            
        return input_features
