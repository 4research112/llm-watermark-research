# Copyright 2024 THU-BPM MarkLLM.
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

# ================================================
# unigram.py
# Description: Implementation of Unigram algorithm
# ================================================

from typing import Union
import torch
import hashlib
import numpy as np
from math import sqrt
from functools import partial
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class UnigramConfig(BaseConfig):
    """Config class for Unigram algorithm, load config file and initialize parameters."""

    # def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
    #     """
    #         Initialize the Unigram configuration.

    #         Parameters:
    #             algorithm_config (str): Path to the algorithm configuration file.
    #             transformers_config (TransformersConfig): Configuration for the transformers model.
    #     """
    #     if algorithm_config is None:
    #         config_dict = load_config_file('config/Unigram.json')
    #     else:
    #         config_dict = load_config_file(algorithm_config)
    #     if config_dict['algorithm_name'] != 'Unigram':
    #         raise AlgorithmNameMismatchError('Unigram', config_dict['algorithm_name'])
        
    #     self.gamma = config_dict['gamma']
    #     self.delta = config_dict['delta']
    #     self.hash_key = config_dict['hash_key']
    #     self.z_threshold = config_dict['z_threshold']

    #     self.generation_model = transformers_config.model
    #     self.generation_tokenizer = transformers_config.tokenizer
    #     self.vocab_size = transformers_config.vocab_size
    #     self.device = transformers_config.device
    #     self.gen_kwargs = transformers_config.gen_kwargs
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Unigram'


class UnigramUtils:
    """Utility class for Unigram algorithm, contains helper functions."""

    def __init__(self, config: UnigramConfig, *args, **kwargs) -> None:
        """
            Initialize the Unigram utility class.

            Parameters:
                config (UnigramConfig): Configuration for the Unigram algorithm.
        """
        self.config = config
        self.mask = np.array([True] * int(self.config.gamma * self.config.vocab_size) + 
                             [False] * (self.config.vocab_size - int(self.config.gamma * self.config.vocab_size)))
        self.rng = np.random.default_rng(self._hash_fn(self.config.hash_key))
        self.rng.shuffle(self.mask)
        
    @staticmethod
    def _hash_fn(x: int) -> int:
        """hash function to generate random seed, solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')
    
    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor) -> tuple:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids)
        green_token_count = 0
        green_token_flags = []
        for idx in range(0, len(input_ids)):
            curr_token = input_ids[idx]
            if self.mask[curr_token] == True:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        print(f"N: {num_tokens_scored}, NG: {green_token_count}")
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class UnigramLogitsProcessor(LogitsProcessor):
    """Logits processor for Unigram algorithm."""

    def __init__(self, config: UnigramConfig, utils: UnigramUtils, *args, **kwargs):
        """
            Initialize the Unigram logits processor.

            Parameters:
                config (UnigramConfig): Configuration for the Unigram algorithm.
                utils (UnigramUtils): Utility class for the Unigram algorithm.
        """
        self.config = config
        self.utils = utils
        self.green_list_mask = torch.tensor(self.utils.mask, dtype=torch.float32)

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the logits for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits and add watermark."""
        greenlist_mask = torch.zeros_like(scores)
        for i in range(input_ids.shape[0]):
            greenlist_mask[i] = self.green_list_mask
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta)
        return scores


class Unigram(BaseWatermark):
    """Top-level class of Unigram algorithm"""

    def __init__(self, algorithm_config: str | UnigramConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the Unigram algorithm.

            Parameters:
                algorithm_config (str | UnigramConfig): Path to the algorithm configuration file or UnigramConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = UnigramConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, UnigramConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a UnigramConfig instance")
        
        self.utils = UnigramUtils(self.config)
        self.logits_processor = UnigramLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the prompt."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Isolate newly generated tokens by excluding the prompt tokens
        new_tokens = encoded_watermarked_text[:, encoded_prompt["input_ids"].shape[-1]:]
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return watermarked_text
        
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[tuple, dict]:
        """Detect watermark in the given text."""

        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # compute z_score
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
    
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)