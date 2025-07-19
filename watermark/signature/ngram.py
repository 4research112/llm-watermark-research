# =============================================
# ngram.py
# Description: n-gram signature collection and detection
# =============================================

import json
from typing import Optional, Set, List, Dict, Tuple, Any, Union
from visualize.data_for_visualization import DataForVisualization
from watermark.kgw.kgw import KGW
from watermark.sweet.sweet import SWEET
from watermark.unigram.unigram import Unigram
from watermark.signature.signature import SignatureSetCollector, KGWSignature, SweetSignature, UnigramSignature, WatermarkTokenAnalyzer
import os
import torch


class NGramSignatureSetUtils:
    @staticmethod
    def load(file_path: str) -> Tuple[int, Set[Tuple[int, ...]]]:
        """
        Load n-gram signature set from file.
        
        Args:
            file_path: signature set file path
            
        Returns:
            Tuple[int, Set[Tuple[int, ...]]]: n value and signature set
            
        Raises:
            FileNotFoundError: if file does not exist
            json.JSONDecodeError: if JSON format is incorrect
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            n = data.get("n", 3)  # default n is 3
            ngram_signature_set = {tuple(ngram) for ngram in data.get("signatures", [])}
            
            print(f"Loaded {len(ngram_signature_set)} {n}-gram signatures from {file_path}")
            return n, ngram_signature_set
        except FileNotFoundError:
            raise FileNotFoundError(f"File does not exist: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON format error: {e.msg}", e.doc, e.pos)
    
    @staticmethod
    def save(ngram_signature_set: Set[Tuple[int, ...]], n: int, save_path: str) -> None:
        """
        Save n-gram signature set to file.
        
        Args:
            ngram_signature_set: n-gram signature set
            n: n-gram value
            save_path: save path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert tuple to list that can be serialized
        saveable_signatures = [list(ngram) for ngram in ngram_signature_set]
        
        with open(save_path, 'w') as f:
            json.dump({
                "n": n,
                "signatures": saveable_signatures
            }, f)
        print(f"Saved {len(ngram_signature_set)} {n}-gram signatures to {save_path}")


class NGramSignatureSetCollector(SignatureSetCollector):
    """
    A tool class for collecting and managing n-gram signature sets.
    
    Collects consecutive "red" tokens in generative watermarking for later detection to improve accuracy.
    """
    
    def __init__(self, watermark, n=3) -> None:
        """
        Initialize n-gram signature collector.
        
        Args:
            watermark: watermark system instance, used to get greenlist and other information
            n: minimum length of consecutive red tokens
        """
        super().__init__(watermark)
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()  # store n-gram signatures
    
    def collect_from_text(self, text: str) -> None:
        """
        Collect consecutive red tokens from a single text that meet n-gram conditions.
        
        Args:
            text: text to analyze
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._collect_ngram_from_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._collect_ngram_from_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._collect_ngram_from_unigram(encoded_text)
        else:
            raise NotImplementedError(f"Unsupported watermark type: {type(self.watermark).__name__}")
    
    def _collect_ngram_from_kgw(self, encoded_text: torch.LongTensor) -> None:
        """Collect consecutive red tokens from KGW watermark text that meet n-gram conditions."""
        # 1. Mark each position as red or green
        red_flags = []
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            red_flags.append(curr_token not in greenlist_ids)
        
        # 2. Collect consecutive n or more red token sequences
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # If it's a red token, add it to the current sequence
                current_seq.append(encoded_text[idx + self.prefix_length].item())
                
                # If the sequence length reaches n, extract a new n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # Take the last n elements
                    self.ngram_signature_set.add(ngram)
            else:
                # When encountering a green token, reset the sequence
                current_seq = []
    
    def _collect_ngram_from_sweet(self, encoded_text: torch.LongTensor) -> None:
        """
        Collect consecutive red tokens from SWEET watermark text that meet n-gram conditions.
        
        SWEET's red token determination requires both:
        1. Not in the greenlist
        2. Entropy value higher than threshold
        """
        # 1. Calculate entropy
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        # 2. Mark each position as red or green
        red_flags = []
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Check if entropy is higher than threshold
            is_high_entropy = idx < len(entropy_list) and entropy_list[idx] > self.watermark.config.entropy_threshold
            
            # Both must be true: not in greenlist and entropy is high
            red_flags.append(curr_token not in greenlist_ids and is_high_entropy)
        
        # 3. Collect consecutive n or more red token sequences
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # If it's a red token, add it to the current sequence
                current_seq.append(encoded_text[idx + self.prefix_length].item())
                
                # If the sequence length reaches n, extract a new n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # Take the last n elements
                    self.ngram_signature_set.add(ngram)
            else:
                # When encountering a green token, reset the sequence
                current_seq = []
    
    def _collect_ngram_from_unigram(self, encoded_text: torch.LongTensor) -> None:
        """Collect consecutive red tokens from Unigram watermark text that meet n-gram conditions."""
        # 1. Mark each position as red or green
        red_flags = []
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            red_flags.append(not self.watermark.utils.mask[curr_token])
        
        # 2. Collect consecutive n or more red token sequences
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # If it's a red token, add it to the current sequence
                current_seq.append(encoded_text[idx].item())
                
                # If the sequence length reaches n, extract a new n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # Take the last n elements
                    self.ngram_signature_set.add(ngram)
            else:
                # When encountering a green token, reset the sequence
                current_seq = []
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """Save n-gram signature set to file"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def load_ngram_signature_set(self, file_path: str) -> None:
        """Load n-gram signature set from file"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)


class KGWNGramSignature(KGWSignature):  
    """
    KGW watermark with n-gram signature-aware detection.
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize n-gram signature-aware KGW watermark.
        
        Args:
            algorithm_config: algorithm configuration file path or configuration object
            transformers_config: Transformers configuration
            n: minimum length of consecutive red tokens
            signature_set: signature set
            signature_file: signature file path
            ngram_signature_set: n-gram signature set
            ngram_signature_file: n-gram signature file path
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)

    def load_ngram_signature_set(self, file_path: str) -> None:
        """Load n-gram signature set from file"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """Save n-gram signature set to file"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """Detect watermark using n-gram rules"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        Apply n-gram consecutive red token rules for scoring.
        
        Args:
            input_ids: encoded text tensor
        
        Returns:
            Tuple[float, List[int]]: z-score and token flags
            Token flags: -1 in prefix or signature, 1 in greenlist, 0 in red
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. Mark all tokens as green
        token_flags = [1] * len(input_ids)
        
        # 2. Mark prefix as -1
        prefix_length = self.config.prefix_length  # Get prefix_length from config
        for i in range(min(prefix_length, len(input_ids))):
            token_flags[i] = -1
        
        # 3. Check if each possible n-gram sequence matches the complete signature
        for i in range(prefix_length, len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # Found complete signature, mark the entire sequence as -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 4. For tokens not marked as signature, determine red or green based on greenlist
        for i in range(prefix_length, len(input_ids)):
            if token_flags[i] != -1:  # If it's not a signature
                # Get greenlist at current position
                greenlist = self.utils.get_greenlist_ids(input_ids[:i])
                curr_token = input_ids[i].item()
                token_flags[i] = 1 if curr_token in greenlist else 0
        
        # 5. Calculate z-score (only consider tokens not in prefix or signature)
        green_count = sum(1 for flag in token_flags[prefix_length:] if flag == 1)
        valid_count = sum(1 for flag in token_flags[prefix_length:] if flag != -1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, highlight_values = self.ngram_score_sequence(encoded_text)

        print(f'z_score: {z_score}, highlight_values: {highlight_values}, len(highlight_values): {len(highlight_values)}')
        red_ratio = sum(1 for value in highlight_values if value == 0) / len(highlight_values)
        print(f'red_ratio: {red_ratio:.2f}')
        green_ratio = sum(1 for value in highlight_values if value == 1) / len(highlight_values)
        print(f'green_ratio: {green_ratio:.2f}')
        ignore_ratio = sum(1 for value in highlight_values if value == -1) / len(highlight_values)
        print(f'ignore_ratio: {ignore_ratio:.2f}')
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)    
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        Return n-gram signature set size.
        
        Returns:
            int: number of sequences in n-gram signature set
        """
        return len(self.ngram_signature_set)
    
class SweetNGramSignature(SweetSignature):
    """
    SWEET watermark with n-gram signature-aware detection.
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize n-gram signature-aware SWEET watermark.
        
        Args:
            algorithm_config: algorithm configuration file path or configuration object
            transformers_config: Transformers configuration
            n: minimum length of consecutive red tokens
            signature_set: signature set
            signature_file: signature file path
            ngram_signature_set: n-gram signature set
            ngram_signature_file: n-gram signature file path
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)
    
    def load_ngram_signature_set(self, file_path: str) -> None:
        """Load n-gram signature set from file"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """Save n-gram signature set to file"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """Detect watermark using n-gram rules"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        Apply n-gram consecutive red token rules for scoring.
        
        Args:
            input_ids: encoded text tensor
        
        Returns:
            Tuple[float, List[int]]: z-score and token flags
            Token flags: -1 in prefix or signature, 1 in greenlist, 0 in red
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. Mark all tokens as green
        token_flags = [1] * len(input_ids)
        
        # 2. Mark prefix as -1
        prefix_length = self.config.prefix_length  # Get prefix_length from config
        for i in range(min(prefix_length, len(input_ids))):
            token_flags[i] = -1
        
        # 3. Calculate entropy
        entropy_list = self.utils.calculate_entropy(
            self.config.generation_model, 
            input_ids
        )
        
        # 4. Check if each possible n-gram sequence matches the complete signature
        for i in range(prefix_length, len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # Found complete signature, mark the entire sequence as -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 5. For tokens not marked as signature, determine red or green based on greenlist and entropy
        for i in range(prefix_length, len(input_ids)):
            if token_flags[i] != -1:  # If it's not a signature
                # Check entropy
                is_high_entropy = i < len(entropy_list) and entropy_list[i] > self.config.entropy_threshold
                
                if not is_high_entropy:
                    # Low entropy tokens do not participate in watermark detection, marked as -1
                    token_flags[i] = -1
                    continue
                    
                # For high entropy tokens, get greenlist at current position and determine red or green
                greenlist = self.utils.get_greenlist_ids(input_ids[:i])
                curr_token = input_ids[i].item()
                
                # In greenlist is green, otherwise red
                token_flags[i] = 1 if curr_token in greenlist else 0
        
        # 6. Calculate z-score (only consider tokens with high entropy and not in prefix or signature)
        green_count = sum(1 for flag in token_flags if flag == 1)
        valid_count = sum(1 for flag in token_flags if flag == 0 or flag == 1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        Return n-gram signature set size.
        
        Returns:
            int: number of sequences in n-gram signature set
        """
        return len(self.ngram_signature_set)
    
class UnigramNGramSignature(UnigramSignature): 
    """
    Unigram watermark with n-gram signature-aware detection.
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize n-gram signature-aware Unigram watermark.
        
        Args:
            algorithm_config: algorithm configuration file path or configuration object
            transformers_config: Transformers configuration
            n: minimum length of consecutive red tokens
            signature_set: signature set
            signature_file: signature file path
            ngram_signature_set: n-gram signature set
            ngram_signature_file: n-gram signature file path
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)

    def load_ngram_signature_set(self, file_path: str) -> None:
        """Load n-gram signature set from file"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """Save n-gram signature set to file"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """Detect watermark using n-gram rules"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        Apply n-gram consecutive red token rules for scoring.
        
        Args:
            input_ids: encoded text tensor
        
        Returns:
            Tuple[float, List[int]]: z-score and token flags
            Token flags: -1 in signature, 1 in greenlist, 0 in red
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. Mark all tokens as green
        token_flags = [1] * len(input_ids)
        
        # 2. Check if each possible n-gram sequence matches the complete signature
        for i in range(len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # Found complete signature, mark the entire sequence as -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 3. For tokens not marked as signature, determine red or green based on mask
        for i in range(len(input_ids)):
            if token_flags[i] != -1:  # If it's not a signature
                token_flags[i] = 1 if self.utils.mask[input_ids[i].item()] else 0
        
        # 4. Calculate z-score (only consider tokens not in signature)
        green_count = sum(1 for flag in token_flags if flag == 1)
        valid_count = sum(1 for flag in token_flags if flag != -1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        Return n-gram signature set size.
        
        Returns:
            int: number of sequences in n-gram signature set
        """
        return len(self.ngram_signature_set)

class NGramWatermarkTokenAnalyzer(WatermarkTokenAnalyzer):
    """
    Analyze the distribution of n-gram consecutive red/green tokens in watermark text.
    Inherit from WatermarkTokenAnalyzer.
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram, 'KGWNGramSignature'], n: int = 3) -> None:
        """
        Initialize n-gram analyzer.
        
        Args:
            watermark: watermark system instance, used to determine green and red tokens
            n: n-gram n value
        """
        super().__init__(watermark)
        self.n = n
        # n-gram red and green token sequence counters
        self.ngram_stats: Dict[Tuple[int, ...], Dict[str, int]] = {}
        
    def analyze_text(self, text: str) -> None:
        """
        Analyze the number of green and red tokens for each token in the text, and analyze the n-gram sequence.
        
        Args:
            text: watermark text
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW) or hasattr(self.watermark, 'ngram_utils'):
            self._analyze_kgw_with_ngram(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._analyze_sweet_with_ngram(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._analyze_unigram_with_ngram(encoded_text)
        else:
            raise NotImplementedError(f"Unsupported watermark type: {type(self.watermark).__name__}")
    
    def _analyze_kgw_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        Analyze green and red tokens in KGW watermark text, and analyze n-gram
        
        Args:
            encoded_text: encoded text tensor
        """
        # Mark each position as red or green
        token_labels = []  # 1 is green, 0 is red
        tokens = []
        
        # Use parent class method to analyze single token
        self._analyze_kgw(encoded_text)
        
        # Collect token sequences and labels
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Determine if it's in greenlist
            if curr_token in greenlist_ids:
                token_labels.append(1)  # green
            else:
                token_labels.append(0)  # red
        
        # Analyze n-gram of consecutive n tokens
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram label: if all are green, then green n-gram, otherwise red n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def _analyze_sweet_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        Analyze green and red tokens in SWEET watermark text, and analyze n-gram
        
        Args:
            encoded_text: encoded text tensor
        """
        # Mark each position as red or green
        token_labels = []  # 1 is green, 0 is red
        tokens = []
        
        # Use parent class method to analyze single token
        self._analyze_sweet(encoded_text)
        
        # Collect token sequences and labels
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Determine if it's in greenlist
            if curr_token in greenlist_ids:
                token_labels.append(1)  # green
            else:
                token_labels.append(0)  # red
        
        # Analyze n-gram of consecutive n tokens
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram label: if all are green, then green n-gram, otherwise red n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def _analyze_unigram_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        Analyze green and red tokens in Unigram watermark text, and analyze n-gram
        
        Args:
            encoded_text: encoded text tensor
        """
        # Mark each position as red or green
        token_labels = []  # 1 is green, 0 is red
        tokens = []
        
        # Use parent class method to analyze single token
        self._analyze_unigram(encoded_text)
        
        # Collect token sequences and labels
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # Determine if it's in greenlist
            if self.utils.mask[curr_token]:
                token_labels.append(1)  # green
            else:
                token_labels.append(0)  # red
        
        # Analyze n-gram of consecutive n tokens
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram label: if all are green, then green n-gram, otherwise red n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def get_ngram_stats(self) -> List[Dict[str, Any]]:
        """
        Get n-gram statistics.
        
        Returns:
            List[Dict]: list of n-gram token sequences, green count, and red count
        """
        result = []
        for ngram, counts in self.ngram_stats.items():
            total_count = counts["green_count"] + counts["red_count"]
            
            # Try to decode n-gram
            try:
                decoded_ngram = "".join([self.tokenizer.decode(token) for token in ngram])
            except:
                decoded_ngram = "<cannot decode>"
                
            result.append({
                "ngram": list(ngram),  # Convert to list for JSON serialization
                "decoded": decoded_ngram,
                "green_count": counts["green_count"],
                "red_count": counts["red_count"],
                "total_count": total_count,
                "green_ratio": counts["green_count"] / total_count if total_count > 0 else 0
            })
        
        # Sort by total count
        result.sort(key=lambda x: x["total_count"], reverse=True)
        return result
    
    def save_ngram_stats(self, save_path: str) -> None:
        """
        Save n-gram statistics to JSON file.
        
        Args:
            save_path: save path
        """
        stats = self.get_ngram_stats()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "n": self.n,
                "ngram_stats": stats
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(stats)} {self.n}-gram statistics to {save_path}")
    
    def clear_stats(self) -> None:
        """
        Clear all statistics.
        """
        super().clear_stats()  # Clear parent class's token_stats
        self.ngram_stats.clear()  # Clear n-gram statistics