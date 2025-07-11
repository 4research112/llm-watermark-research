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
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import re
import copy
import nltk
import torch
import random
import numpy as np
from tqdm import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import DiversityValueError
from evaluation.tools.oracle import QualityOracle
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text

class RandomWalkAttack(TextEditor):
    """
        Remove the watermark using the random walk attack (https://arxiv.org/abs/2311.04378) via black-box access to a quality oracle and a perturbaiton oracle.
        (1) Quality oracle can evaluate whether a candidate output is a high-quality response to a prompt.
        (2) Perturbation oracle can modify an output with a nontrivial probability of maintaining quality, 
            and which induces an efficiently mixing random walk on high-quality outputs.
        
        Examplar Usage: 
        '''
        model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto') 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xl", device_map='auto')
        perturbation_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
        quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
        span_length = 6
        attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                  quality_oracle=quality_oracle,
                                  max_new_tokens=int(2*span_length), min_length=int(1.5*span_length), 
                                  do_sample=True, top_p=0.95, top_k=None, repetition_penalty=1.5)
        '''
    """

    def __init__(self, perturbation_tokenizer: T5Tokenizer, perturbation_oracle: T5ForConditionalGeneration, quality_oracle: QualityOracle,
                       device='cuda', total_steps=200, span_len=6, target_valid_steps=100, **kwargs):
        """
            Parameters:
            perturbation_tokenizer (T5Tokenizer): The tokenizer for the perturbation oracle.
            perturbation_oracle (T5ForConditionalGeneration): The perturbation oracle.
            quality_oracle (QualityOracle): The quality oracle.
            device (str): The device to use for inference.
            span_len (int): The length of the span to mask in each random walk step.
            total_steps (int): The total number of random walk steps.
            target_valid_steps (int): The target number of valid steps.
        """
        self.perturbation_tokenizer = perturbation_tokenizer
        self.perturbation_oracle = perturbation_oracle.eval()
        self.quality_oracle = quality_oracle
        self.device = device
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)
        
        self.span_len = span_len
        self.total_steps = total_steps
        self.target_valid_steps = target_valid_steps
        if self.quality_oracle.check_quality == 'checker':
            from gramformer import Gramformer
            self.gf = Gramformer(models = 1, use_gpu=True)

    def perturb(self, text: str):
        final_input_text = self.mask_text(text)

        # Tokenize the input
        final_input = self.perturbation_tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the edited text
        with torch.inference_mode():
            outputs = self.perturbation_oracle.generate(**final_input, **self.gen_kwargs)
        outputs = self.perturbation_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        infilled_text = outputs[0]
        final_output_text = final_input_text.replace('<extra_id_0>', infilled_text)

        return final_output_text

    def edit(self, text: str, prompt: str, backtrack_patience: int = 100, max_attempts: int = 1000):
        """Edit the text using the T5 model."""

        original_response, n_response = text, text
        n_iter, valid_steps = 0, 0
        patience = 0
        cached_response = copy.deepcopy(n_response)
        # Process the input text in sentence windows
        pbar = tqdm(total=None)
        while n_iter < self.total_steps or valid_steps < self.target_valid_steps:
            candidate_response = self.perturb(n_response)

            candidate_response = self.grammatical_error_correction(candidate_response)
            candidate_response = self.remove_incomplete_sentences(candidate_response)
            
            if self.quality_oracle.maintain_quality(prompt, original_response, candidate_response):
                cached_response = n_response
                n_response = candidate_response
                valid_steps += 1
                if valid_steps % 10 == 0:
                    print(f"Original response: {original_response}")
                print(f"Get a better {valid_steps}-th response at step {n_iter}/{self.total_steps}: {n_response}")
                patience = 0
            else:
                patience += 1
            
            if patience > max_attempts:
                break
            elif patience > backtrack_patience:
                n_response = cached_response
                patience = 0
            
            pbar.update(1)
            n_iter += 1
        pbar.close()

        return n_response

    def grammatical_error_correction(self, text):
        sentences = sent_tokenize(text)
        corrected_sents = []
        for sent in sentences:
            corrected_sent = self.gf.correct(sent, max_candidates=1).pop()
            corrected_sents.append(corrected_sent)
        corrected_text = ' '.join(corrected_sents)
        return corrected_text

    def mask_text(self, text):
        words = text.replace('\n', ' \n').split(' ')
        if len(words) == 1:
            return text + ' <extra_id_0> '
        start = np.random.randint(0, len(words) - self.span_len)
        end = start + self.span_len
        masked_text = ' '.join(words[:start]) + ' <extra_id_0> ' + ' '.join(words[end:])
        return masked_text
    
    def contains_verb(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        return any(tag.startswith('VB') for word, tag in tagged_words)

    def remove_incomplete_sentences(self, text):
        sentences = sent_tokenize(text)
        complete_sentences = []
        for sent in sentences:
            if sent.endswith('.') and not self.contains_verb(sent) and not bool(re.match(r'^\d+\.$', sent)):
                continue
            else:
                complete_sentences.append(sent)
        return ' '.join(complete_sentences)

    def correct_text(self, text):
        """Basic punctuation correction"""
        # Replace multiple spaces with a single space
        corrected_text = re.sub(r'\s+', ' ', text)

        # Correct spaces before commas, periods, colons, semicolons, exclamation marks, and question marks
        corrected_text = re.sub(r'\s+([,.;!?])', r'\1', corrected_text)  # Remove space before punctuation
        corrected_text = re.sub(r'([,.;!?])(?!\s)', r'\1 ', corrected_text)  # Ensure space after punctuation if missing

        # Replace multiple occurrences of punctuation marks with a single instance
        # This part targets specific punctuation marks (you can add more as needed)
        corrected_text = re.sub(r'(\.){2,}', '.', corrected_text)
        corrected_text = re.sub(r'(,){2,}', ',', corrected_text)
        corrected_text = re.sub(r'(!){2,}', '!', corrected_text)
        corrected_text = re.sub(r'(\?){2,}', '?', corrected_text)
        corrected_text = re.sub(r'(:){2,}', ':', corrected_text)
        corrected_text = re.sub(r'(;){2,}', ';', corrected_text)

        return corrected_text

class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content="Your are a helpful assistant to rewrite the text.")
        paraphrased_text = openai_util.get_result(self.prompt + text)
        return paraphrased_text

class TaideParaphraser(TextEditor):
    """使用 TAIDE 模型進行文本改寫的類。"""

    def __init__(self, tokenizer, model, transformers_config, prompt: str = "請重寫以下文字，保持原意但使用不同表達方式："):
        """
        初始化 TAIDE 文本改寫器。

        參數:
            tokenizer: TAIDE 模型的 tokenizer
            model: TAIDE 模型
            prompt (str): 用於改寫文本的提示詞
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """
        使用 TAIDE 模型改寫文本。

        參數:
            text (str): 需要改寫的文本
            reference (str, optional): 參考文本，默認為 None

        返回:
            str: 改寫後的文本
        """
        self.gen_kwargs['temperature'] = 0.2
        final_input_text = f"{self.prompt}\n{text}"
        encoded_prompt = self.tokenizer(final_input_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        # Generate unwatermarked text
        encoded_unwatermarked_text = self.model.generate(**encoded_prompt, **self.gen_kwargs)
        # Isolate newly generated tokens by excluding the prompt tokens
        new_tokens = encoded_unwatermarked_text[:, encoded_prompt["input_ids"].shape[-1]:]
        # Decode
        paraphrased_text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
        return paraphrased_text

class DipperParaphraser(TextEditor):
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")
    
    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)
        
        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        
        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())
        
        output_text = ""
        
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])
            
            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            
            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            
            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text


class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the prompt from the text."""
        if reference is not None:
            truncated_text = ' '.join(text.split()[len(reference.split()):])
            return truncated_text
        else:
            return text


class TruncateTaskTextEditor(TextEditor):
    """Truncate the task description from the text, used in code generation."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the task description from the text."""
        if reference is not None:
            truncated_text = text[len(reference):]
            return truncated_text
        else:
            return text
        

class CodeGenerationTextEditor(TextEditor):
    """Process the code generation output, removing the extra parts."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Process the code generation output, removing the extra parts."""
        text = text.lstrip("\n")
        text = text.split("\n\n")[0]
        return text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                 translate_to_source = Translator(from_lang="zh", to_lang="en").translate) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result

class ScrambleAttack(TextEditor):

    def __init__(self, tokenizer=None, min_length: int = 0) -> None:
        """
        Initialize the scramble attack.

        Parameters:
            tokenizer: The tokenizer to use for the attack.
            min_length: The minimum length of the text to attack.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.min_length = min_length

    def edit(self, text: str, reference=None):
        """Scramble the text."""

        # Check if the text length meets the attack condition
        if self.tokenizer and self.min_length > 0:
            token_length = len(self.tokenizer(text)["input_ids"])
            if token_length < self.min_length:
                return ""
        
        # Split the text by periods and shuffle the sentences
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return text
        
        # Shuffle the sentences randomly
        random.shuffle(sentences)
        
        # Reassemble the text
        scrambled_text = ". ".join(sentences)
        if not scrambled_text.endswith("."):
            scrambled_text += "."
            
        return scrambled_text

class CopyPasteAttack(TextEditor):
    """Copy-Paste attack: copy and paste a part of the reference text into the target text."""

    def __init__(self, tokenizer, 
                 num_insertions: int = 3, 
                 insertion_length: int = 20,
                 min_length: int = 0,
                 attack_type: str = "k-t") -> None:
        """
        Initialize the copy-paste attack.

        Parameters:
            tokenizer: The tokenizer to use for the attack.
            num_insertions (int): The number of insertions (k).
            insertion_length (int): The length of each insertion (t).
            min_length (int): The minimum length of the text to attack.
            attack_type (str): The attack type, supports "single-single", "triple-single", "k-t".
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.num_insertions = num_insertions
        self.insertion_length = insertion_length
        self.min_length = min_length
        self.attack_type = attack_type

    def _tokenize_text(self, text: str):
        """Convert the text to token IDs."""
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def _single_insertion(self, insertion_len: int, min_token_count: int, 
                         dst_tokens, src_tokens):
        """Execute single insertion attack."""
        # Ensure the insertion length does not exceed the available length
        actual_insertion_len = min(insertion_len, min_token_count - 1)
        
        # Randomly select the insertion position
        insertion_pos = random.randint(1, len(dst_tokens) - actual_insertion_len)
        
        # Randomly select the starting position of the source text
        src_start = random.randint(0, len(src_tokens) - actual_insertion_len)
        
        # Execute insertion: dst前部 + src片段 + dst後部
        result = torch.cat([
            dst_tokens[:insertion_pos],
            src_tokens[src_start:src_start + actual_insertion_len],
            dst_tokens[insertion_pos + actual_insertion_len:]
        ])
        
        return result

    def _k_insertion_t_len(self, k: int, t: int, min_token_count: int,
                          dst_tokens, src_tokens):
        """執行 k 次長度為 t 的「定點位置對應替換」攻擊。"""
        
        # 確保文本長度足夠進行攻擊
        if min_token_count < t * k:
            return dst_tokens.clone()

        while True:
            # 直接在有效的範圍內生成隨機起始點，確保片段不會超出邊界
            try:
                rand_insert_locs = torch.randperm(min_token_count - t)[:k]
            except RuntimeError:
                # 如果可選的起始點數量小於 k，則無法攻擊
                return dst_tokens.clone()
            
            # 排序位置，方便檢查重疊
            sorted_locs, _ = torch.sort(rand_insert_locs)
            
            # 檢查重疊條件
            overlap = False
            for i in range(len(sorted_locs) - 1):
                if sorted_locs[i] + t > sorted_locs[i+1]:
                    overlap = True
                    break
            
            # 如果沒有重疊，則找到了有效的位置組合
            if not overlap:
                break
        
        # 執行替換操作
        result_tokens = dst_tokens.clone()
        
        for loc in sorted_locs:
            start_idx = loc.item()
            end_idx = start_idx + t
            
            # 核心邏輯：用源文本的對應位置替換目標文本
            result_tokens[start_idx:end_idx] = src_tokens[start_idx:end_idx]
        
        return result_tokens

    def edit(self, text: str, reference: str = None):
        """
        執行 Copy-Paste 攻擊。

        參數:
            text (str): 目標文本
            reference (str): 源文本

        返回:
            str: 攻擊後的文本
        """
        if reference is None:
            return text
        
        # Tokenize 兩個文本
        dst_tokens = self._tokenize_text(reference)
        src_tokens = self._tokenize_text(text)
        
        # 檢查長度條件
        if self.min_length > 0:
            if len(dst_tokens) < self.min_length or len(src_tokens) < self.min_length:
                return ""
        
        min_token_count = min(len(dst_tokens), len(src_tokens))
        
        # 根據攻擊類型執行不同的攻擊
        if self.attack_type == "single-single":
            attacked_tokens = self._single_insertion(
                self.insertion_length, min_token_count, dst_tokens, src_tokens
            )
        elif self.attack_type == "k-t":
            attacked_tokens = self._k_insertion_t_len(
                self.num_insertions, self.insertion_length, 
                min_token_count, dst_tokens, src_tokens
            )
        else:
            raise ValueError(f"不支援的攻擊類型: {self.attack_type}")
        
        # 將 tokens 轉換回文本
        attacked_text = self.tokenizer.decode(attacked_tokens, skip_special_tokens=True)
        
        return attacked_text


class PercentageCopyPasteAttack(CopyPasteAttack):
    """支援小數比例形式的 Copy-Paste 攻擊。"""

    def __init__(self, tokenizer, 
                 num_insertions: int = 3,
                 insertion_ratio: float = 0.25,
                 max_new_tokens: int = 200,
                 min_length: int = 0,
                 attack_type: str = "k-t") -> None:
        """
        初始化支援小數比例的 Copy-Paste 攻擊。

        參數:
            tokenizer: 用於 tokenization 的 tokenizer
            num_insertions (int): 插入次數
            insertion_ratio (float): 插入比例，如 0.25 表示 25%
            max_new_tokens (int): 最大新 token 數，用於比例計算
            min_length (int): 最小文本長度
            attack_type (str): 攻擊類型
        """
        # 驗證比例值的合理性
        if not 0.0 <= insertion_ratio <= 1.0:
            raise ValueError(f"insertion_ratio 必須在 0.0 到 1.0 之間，當前值: {insertion_ratio}")
        
        # 計算實際插入長度
        insertion_length = int(insertion_ratio * max_new_tokens) // num_insertions
        
        super().__init__(tokenizer, num_insertions, insertion_length, min_length, attack_type)
        
        self.insertion_ratio = insertion_ratio
        self.max_new_tokens = max_new_tokens
        
        # 計算有效攻擊百分比（用於顯示）
        self.effective_attack_percentage = (1 - insertion_ratio) * 100