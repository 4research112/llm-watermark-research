# =============================================
# signature.py
# Description: Signature collection and signature-aware watermarking
# =============================================

import os
import json
from typing import Optional, Set, List, Dict, Tuple, Any, Union
import torch
from watermark.ewd.ewd import EWD
from watermark.kgw.kgw import KGW
from watermark.sweet.sweet import SWEET
from watermark.unigram.unigram import Unigram

class SignatureSetUtils:
    @staticmethod
    def load(file_path: str) -> Set[int]:
        """Load signature set from file"""
        try:
            with open(file_path, 'r') as f:
                signature_set = set(json.load(f))
            print(f"Loaded {len(signature_set)} signatures from {file_path}")
            return signature_set
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON format error: {e.msg}", e.doc, e.pos)
    
    @staticmethod
    def save(signature_set: Set[int], save_path: str) -> None:
        """Save signature set to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(list(signature_set), f)
        print(f"Saved {len(signature_set)} signatures to {save_path}")

class SignatureSetCollector:
    """
    A utility class for collecting and managing signature sets.
    
    Collects "red" tokens from generative watermarking for later detection accuracy.
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram]) -> None:
        """
        Initialize signature collector.
        
        Args:
            watermark: watermark system instance, used to get greenlist and other information
        """
        self.watermark = watermark
        self.signature_set: Set[int] = set()
        self.tokenizer = watermark.config.generation_tokenizer
        self.prefix_length = getattr(watermark.config, 'prefix_length', 0)
        self.device = watermark.config.device
        
    def collect_from_text(self, text: str) -> None:
        """
        Collect red tokens from single text.
        
        Args:
            text: text to analyze
        
        Raises:
            NotImplementedError: if watermark type is not supported
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._collect_from_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._collect_from_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._collect_from_unigram(encoded_text)
        else:
            raise NotImplementedError(f"Unsupported watermark type: {type(self.watermark).__name__}")
    
    def _collect_from_kgw(self, encoded_text: torch.LongTensor) -> None:
        """
        Collect red tokens from KGW watermark text.
        
        Args:
            encoded_text: encoded text tensor
        """
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            # If not in greenlist, it's red, add to signature_set
            if curr_token not in greenlist_ids:
                self.signature_set.add(curr_token)
    
    def _collect_from_sweet(self, encoded_text: torch.LongTensor) -> None:
        """
        Collect high-entropy red tokens from SWEET watermark text.
        
        Args:
            encoded_text: encoded text tensor
        """
        # Calculate entropy
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        # Collect high-entropy red tokens
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Check if entropy is higher than threshold
            is_high_entropy = entropy_list[idx] > self.watermark.config.entropy_threshold
            
            # If not in greenlist and entropy is high, it's the signature we want to collect
            if curr_token not in greenlist_ids and is_high_entropy:
                self.signature_set.add(curr_token)
    
    def _collect_from_unigram(self, encoded_text: torch.LongTensor) -> None:
        """
        Collect red tokens from Unigram watermark text.
        
        Args:
            encoded_text: encoded text tensor
        """
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            # Check if in greenlist (i.e. mask value is True)
            if not self.watermark.utils.mask[curr_token]:
                # If not in greenlist, it's red, add to signature_set
                self.signature_set.add(curr_token)
    
    def collect_from_file(self, file_path: str) -> None:
        """
        Collect signatures from file.
        
        Args:
            file_path: text file path
        
        Raises:
            FileNotFoundError: if file not found
            IOError: if error reading file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.collect_from_text(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading file: {e}")
    
    def save_signature_set(self, save_path: str) -> None:
        """Save signature set to file"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def load_signature_set(self, file_path: str) -> None:
        """Load signature set from file"""
        self.signature_set = SignatureSetUtils.load(file_path)

class KGWSignature(KGW):
    """KGW watermark with signature awareness, can exclude tokens in signature set during detection."""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize KGW watermark with signature awareness.
        
        Args:
            algorithm_config: algorithm config file path or config object
            transformers_config: Transformers config
            signature_set: signature set
            signature_file: signature file path
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """Load signature set from file"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """Save signature set to file"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        Override detection method, consider signature set.
        
        Args:
            text: text to detect
            return_dict: whether to return dictionary format result
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: detection result
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Filter out tokens in signature set
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        Consider signature scoring method, exclude tokens in signature set.
        
        Args:
            input_ids: encoded text tensor
        
        Returns:
            Tuple[float, List[int]]: z-score and green token flags
        """
        valid_positions = []
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        filtered_count = 0  # Record number of filtered tokens
        
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # If token in signature set, skip
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # Mark as not counted
                filtered_count += 1
                continue
            
            valid_positions.append(idx)
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        # Calculate number of tokens actually scored
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags  # Too few tokens to score
        
        # Use utils' _compute_z_score function to calculate z-score
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags
    
    @property
    def signature_set_size(self) -> int:
        """
        Return signature set size.
        
        Returns:
            int: number of tokens in signature set
        """
        return len(self.signature_set)


class SweetSignature(SWEET):
    """SWEET watermark with signature awareness, can exclude tokens in signature set during detection."""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize SWEET watermark with signature awareness.
        
        Args:
            algorithm_config: algorithm config file path or config object
            transformers_config: Transformers config
            signature_set: signature set
            signature_file: signature file path
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """Load signature set from file"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """Save signature set to file"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        Override detection method, consider signature set.
        
        Args:
            text: text to detect
            return_dict: whether to return dictionary format result
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: detection result
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # Filter out tokens in signature set
        z_score, green_tokens, weights = self.signature_score_sequence(encoded_text, entropy_list)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor, entropy_list: List[float]) -> Tuple[float, List[int], List[int]]:
        """
        Consider signature scoring method, exclude tokens in signature set.
        
        Args:
            input_ids: encoded text tensor
            entropy_list: 文本中每個token的熵值列表
        
        Returns:
            Tuple[float, List[int], List[int]]: z-score, green token flags, and weights
        """
        # Initialize token flags
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]
        
        # Process each token
        valid_positions = []
        green_token_count = 0
        
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # Get greenlist ID
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            
            # First, determine weight based on entropy
            # This is consistent with the original logic: entropy high is 1, otherwise 0
            if entropy_list[idx] > self.config.entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)
            
            # If token in signature set, mark as -1 and skip scoring
            if curr_token in self.signature_set:
                green_token_flags.append(-1)
                weights[-1] = -1  # Weight of token in signature set is -1
                continue
            
            # Process non-signature set tokens
            if entropy_list[idx] > self.config.entropy_threshold:
                valid_positions.append(idx)
                
                # Check if in greenlist
                if curr_token in greenlist_ids:
                    green_token_flags.append(1)
                    green_token_count += 1
                else:
                    green_token_flags.append(0)
            else:
                # Low entropy tokens are not counted in green token statistics
                green_token_flags.append(-1)
        
        # Calculate z-score
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags, weights
        
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags, weights
    
    @property
    def signature_set_size(self) -> int:
        """
        Return signature set size.
        
        Returns:
            int: number of tokens in signature set
        """
        return len(self.signature_set)


class UnigramSignature(Unigram):
    """Unigram watermark with signature awareness, can exclude tokens in signature set during detection."""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize Unigram watermark with signature awareness.
        
        Args:
            algorithm_config: algorithm config file path or config object
            transformers_config: Transformers config
            signature_set: signature set
            signature_file: signature file path
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """Load signature set from file"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """Save signature set to file"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        Override detection method, consider signature set.
        
        Args:
            text: text to detect
            return_dict: whether to return dictionary format result
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: detection result
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Filter out tokens in signature set
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        Consider signature scoring method, exclude tokens in signature set.
        
        Args:
            input_ids: encoded text tensor
        
        Returns:
            Tuple[float, List[int]]: z-score and green token flags
        """
        valid_positions = []
        green_token_count = 0
        green_token_flags = []
        
        filtered_count = 0  # Record number of filtered tokens
        
        for idx in range(len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # If token in signature set, skip
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # Mark as not counted
                filtered_count += 1
                continue
            
            valid_positions.append(idx)
            if self.utils.mask[curr_token] == True:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        # Calculate number of tokens actually scored
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags  # Too few tokens to score
        
        # Use utils' _compute_z_score function to calculate z-score
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags
    
    @property
    def signature_set_size(self) -> int:
        """
        Return signature set size.
        
        Returns:
            int: number of tokens in signature set
        """
        return len(self.signature_set)
    
class EWDSignature(EWD):
    """EWD watermark with signature awareness, can exclude tokens in signature set during detection."""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize KGW watermark with signature awareness.
        
        Args:
            algorithm_config: algorithm config file path or config object
            transformers_config: Transformers config
            signature_set: signature set
            signature_file: signature file path
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """Load signature set from file"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """Save signature set to file"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        Override detection method, consider signature set.
        
        Args:
            text: text to detect
            return_dict: whether to return dictionary format result
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: detection result
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Filter out tokens in signature set
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor, entropy_list: List[float]) -> Tuple[float, List[int], List[float]]:
        """
        考慮 signature 的評分方法，排除簽名集中的 tokens。
        
        Args:
            input_ids: encoded text tensor
            entropy_list: entropy list of each token in text
        
        Returns:
            Tuple[float, List[int], List[float]]: z-score, green token flags, and weights
        """
        # Check if there are enough tokens to score
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            return 0.0, [], []  # Too few tokens to score
        
        # Initialize green token flags
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        # Initialize weights
        weights = [-1 for _ in range(self.config.prefix_length)]
        
        # Process each token
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # If token in signature set, skip
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # Mark as not counted
                weights.append(-1)  # Weight also marked as not counted
                continue
            
            # Get greenlist ID and determine current token
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            
            # Calculate weight
            if idx >= self.config.prefix_length:
                weights.append(entropy_list[idx])
        
        # Filter out positions marked as -1
        valid_weights = [w for w, f in zip(weights[self.config.prefix_length:], 
                                         green_token_flags[self.config.prefix_length:]) 
                        if f != -1]
        valid_flags = [f for f in green_token_flags[self.config.prefix_length:] 
                      if f != -1]
        
        if not valid_weights:  # If there are no valid weights
            return 0.0, green_token_flags, weights
        
        # Calculate weighted count of green tokens
        green_token_count = sum(w for w, f in zip(valid_weights, valid_flags) if f == 1)
        print(f"signature N: {len(valid_weights)}, signature NG: {green_token_count}")
        # Use utils' _compute_z_score function to calculate z-score
        z_score = self.utils._compute_z_score(green_token_count, valid_weights)
        
        return z_score, green_token_flags, weights
    
    @property
    def signature_set_size(self) -> int:
        """
        Return signature set size.
        
        Returns:
            int: number of tokens in signature set
        """
        return len(self.signature_set)

class WatermarkTokenAnalyzer:
    """
    Analyze red and green token counts in watermark text.
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram]) -> None:
        """
        Initialize analyzer.
        
        Args:
            watermark: watermark system instance, used to determine green and red tokens
        """
        self.watermark = watermark
        self.tokenizer = watermark.config.generation_tokenizer
        self.prefix_length = getattr(watermark.config, 'prefix_length', 0)
        self.device = watermark.config.device
        
        # 紅字和綠字的計數器
        self.token_stats: Dict[int, Dict[str, int]] = {}
        
    def analyze_text(self, text: str) -> None:
        """
        Analyze red and green token counts in each token of text.
        
        Args:
            text: watermark text
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._analyze_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._analyze_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._analyze_unigram(encoded_text)
        else:
            raise NotImplementedError(f"Unsupported watermark type: {type(self.watermark).__name__}")
    
    def _analyze_kgw(self, encoded_text: torch.LongTensor) -> None:
        """Analyze green and red tokens in KGW watermark text"""
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Initialize or update statistics
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # Update count based on whether in greenlist
            if curr_token in greenlist_ids:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def _analyze_sweet(self, encoded_text: torch.LongTensor) -> None:
        """Analyze green and red tokens in SWEET watermark text"""
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # Get greenlist ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # Initialize or update statistics
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # Update count based on whether in greenlist
            if curr_token in greenlist_ids:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def _analyze_unigram(self, encoded_text: torch.LongTensor) -> None:
        """Analyze green and red tokens in Unigram watermark text"""
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # Initialize or update statistics
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # Update count based on whether in greenlist (Unigram directly uses mask)
            if self.watermark.utils.mask[curr_token]:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def analyze_file(self, file_path: str) -> None:
        """
        Read text from file and analyze.
        
        Args:
            file_path: text file path
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.analyze_text(text)
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    def analyze_watermarked_texts_json(self, file_path: str, text_key: str = 'watermarked_text') -> None:
        """
        Analyze token statistics from JSON file containing multiple watermark texts.
        
        Args:
            file_path: JSON file path, each item should contain watermark text
            text_key: key name of watermark text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Error: {file_path} is not a valid text list")
                return
                
            for entry in data:
                if isinstance(entry, dict) and text_key in entry:
                    text = entry[text_key]
                    self.analyze_text(text)
                else:
                    print(f"Warning: text key '{text_key}' not found")
                    
            print(f"Analyzed {len(data)} texts")
        
        except Exception as e:
            print(f"Error analyzing JSON file: {e}")
    
    def get_token_stats(self) -> List[Dict[str, Any]]:
        """
        Get token statistics.
        
        Returns:
            List[Dict]: list of token IDs, green counts, and red counts
        """
        result = []
        for token_id, counts in self.token_stats.items():
            total_count = counts["green_count"] + counts["red_count"]
            result.append({
                "token_id": token_id,
                "green_count": counts["green_count"],
                "red_count": counts["red_count"],
                "total_count": total_count,
                "green_ratio": counts["green_count"] / total_count if total_count > 0 else 0
            })
        
        # Sort by total count
        result.sort(key=lambda x: x["total_count"], reverse=True)
        return result
    
    def save_stats(self, save_path: str) -> None:
        """
        Save token statistics to JSON file.
        
        Args:
            save_path: save path
        """
        stats = self.get_token_stats()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(stats)} token statistics to {save_path}")
    
    def clear_stats(self) -> None:
        """
        Clear all statistics.
        """
        self.token_stats.clear()