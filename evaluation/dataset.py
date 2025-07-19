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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json


class BaseDataset:
    """Base class for dataset."""

    def __init__(self, max_samples: int = 200):
        """
        Initialize the dataset.
        
        Parameters:
            max_samples (int): Maximum number of samples to load. Default is 200.
        """
        self.max_samples = max_samples
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, max_samples: int = 200):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])


class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            # For UnwatermarkedTextDetectionPipeline_V2, we can use canonical_solution as natural_text
            # This represents "human-written code", which can be used as a reference for non-watermarked text
            if 'canonical_solution' in item:
                self.natural_texts.append(item['canonical_solution'])
            else:
                print(f"No canonical solution found for {prompt}")

            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})

class MBPPDataset(BaseDataset):
    """Dataset class for MBPP (Mostly Basic Python Problems) dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
        Initialize the MBPP dataset.

        Parameters:
            data_source (str): The path to the MBPP dataset file.
            max_samples (int): Maximum number of samples to load. Default is 200.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self) -> None:
        """Load data from the MBPP dataset file."""
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            
            # Program problem description as prompt
            self.prompts.append(item['text'])
            
            # Program solution as natural_text
            # Process \r\n line breaks, convert to normal \n
            code = item['code'].replace('\r\n', '\n').replace('\r', '\n')
            self.natural_texts.append(code)
            
            # Optional: save test information to references for later evaluation
            reference_data = {
                'task_id': item['task_id'],
                'text': item['text'],
                'test_list': item['test_list'],
                'test_setup_code': item['test_setup_code']
            }
            if 'challenge_test_list' in item:
                reference_data['challenge_test_list'] = item['challenge_test_list']
            
            self.references.append(reference_data)

if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json', max_samples=100)
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', max_samples=100)
    d3 = HumanEvalDataset('dataset/human_eval/test.jsonl', max_samples=100)
    d4 = MBPPDataset('dataset/mbpp/mbpp.jsonl', max_samples=1000)

    # Get the first question
    prompt = d4.get_prompt(30)  # "Write a function to find the minimum cost path..."
    natural_text = d4.get_natural_text(30)  # "R = 3\nC = 3\ndef min_cost(cost, m, n):..."
    reference = d4.get_reference(30)  # {'task_id': 1, 'text': '...', 'test_list': [...]}
    print(d4.prompt_nums)
    print(d4.natural_text_nums)
    print(d4.reference_nums)
    print(prompt)
    print(natural_text)
    print(reference)