import argparse
from dataclasses import dataclass, field
from typing import Optional, List
from translate import Translator
from evaluation.dataset import C4Dataset, HumanEvalDataset, MBPPDataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType, WMTextDetectionPipeline, WatermarkedTextDetectionPipeline_V2, UnwatermarkedTextDetectionPipeline_V2, SignatureAwareWatermarkDetectionPipeline_V2, SignatureAwareUnwatermarkedTextDetectionPipeline_V2
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor, ScrambleAttack, CopyPasteAttack, PercentageCopyPasteAttack
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setting random seed for reproducibility
seed = 30
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ExperimentParams:
    
    # Basic parameters
    algorithm_name: str
    max_samples: int
    output_dir: str
    dataset_path: str
    # Model selection
    model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'
    # Watermark parameters
    delta: float = 1.0
    temperature: float = 1.0
    # File paths
    watermarked_texts_path: Optional[str] = None
    # Mode settings
    generation_mode: str = "load"
    # Attack parameters
    attack_name: Optional[str] = None
    use_winmax: bool = False
    # n-gram signature config
    n: Optional[int] = None
    # Other parameters
    extract_colors: bool = True
    text_source_mode: str = "natural"
    
    def __post_init__(self):
        """Parameter validation"""
        if self.algorithm_name not in ['KGW', 'SWEET', 'Unigram', 'EXP']:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
        
        if self.generation_mode not in ['load', 'generate']:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}")
            
        supported_models = [
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3.1-8B',
            'facebook/opt-1.3b',
        ]
        if self.model_name not in supported_models:
            print(f"Warning: model {self.model_name} may not be supported, supported models: {supported_models}")
    
    @property
    def is_exp_algorithm(self) -> bool:
        """Whether it's EXP algorithm"""
        return self.algorithm_name == 'EXP'
    
    def get_watermark_param(self) -> dict:
        """Get watermark parameters"""
        if self.is_exp_algorithm:
            return {'temperature': self.temperature}
        else:
            return {'delta': self.delta}
    
    def __repr__(self) -> str:
        return (
            f"ExperimentParams(\n"
            f"  algorithm_name='{self.algorithm_name}',\n"
            f"  model_name='{self.model_name}',\n"
            f"  dataset_path='{self.dataset_path}',\n"
            f"  max_samples={self.max_samples},\n"
            f"  output_dir='{self.output_dir}',\n"
            f"  watermarked_texts_path={self.watermarked_texts_path!r},\n"
            f"  generation_mode='{self.generation_mode}',\n"
            f"  attack_name={self.attack_name!r},\n"
            f"  delta={self.delta},\n"
            f"  temperature={self.temperature},\n"
            f"  use_winmax={self.use_winmax},\n"
            f"  n={self.n},\n"
            f"  extract_colors={self.extract_colors},\n"
            f"  text_source_mode='{self.text_source_mode}'\n"
            f")"
        )

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    path: str
    tokenizer: object
    max_samples: int
    
    def create_dataset(self):
        """Create corresponding dataset based on path"""
        if 'c4' in self.path.lower():
            print(f"Using C4Dataset to load {self.path}")
            return C4Dataset(self.path, max_samples=self.max_samples)
        elif 'human_eval' in self.path.lower():
            print(f"Using HumanEvalDataset to load {self.path}")
            return HumanEvalDataset(self.path, max_samples=self.max_samples)
        elif 'mbpp' in self.path.lower():
            print(f"Using MBPPDataset to load {self.path}")
            return MBPPDataset(self.path, max_samples=self.max_samples)
        else:
            raise ValueError(f"Unsupported dataset type: {self.path}")

def get_transformes_config(model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'):
    """Get transformers configuration"""
    print(f"Using model: {model_name}")

    # OPT model configuration
    if model_name == 'facebook/opt-1.3b':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        vocab_size = 50272
        
    # Llama model configuration
    else:    
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        vocab_size = len(list(tokenizer.get_vocab().values()))

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
        max_new_tokens=200,
        min_length=230,
        no_repeat_ngram_size=4,
        do_sample=True,
        eos_token_id=None,
    )
    return model, tokenizer, transformers_config

def create_watermark(params: ExperimentParams, transformers_config):
    """Create watermark instance"""
    print(f"Initializing {params.algorithm_name} watermark...")
    
    watermark = AutoWatermark.load(
        algorithm_name=params.algorithm_name,
        algorithm_config=f'config/{params.algorithm_name}.json',
        transformers_config=transformers_config,
        **params.get_watermark_param()
    )
    
    if params.is_exp_algorithm:
        print(f"temperature: {watermark.config.temperature}")
    else:
        print(f"delta: {watermark.config.delta}")
    
    print(f"{params.algorithm_name} watermark initialized")
    return watermark

def create_attack(params: ExperimentParams, tokenizer, transformers_config):
    """Create attack instance"""
    if not params.attack_name:
        return None
        
    attack_map = {
        'Word-D': lambda: WordDeletion(ratio=0.3),
        'Word-S': lambda: SynonymSubstitution(ratio=0.5),
        'Word-S-Context': lambda: ContextAwareSynonymSubstitution(
            ratio=0.5,
            tokenizer=BertTokenizer.from_pretrained('bert-large-uncased', local_files_only=True),
            model=BertForMaskedLM.from_pretrained('bert-large-uncased', local_files_only=True).to(device)
        ),
        'scramble': lambda: ScrambleAttack(tokenizer=tokenizer),
        'single-single': lambda: PercentageCopyPasteAttack(
            tokenizer=tokenizer,
            num_insertions=1,
            insertion_ratio=0.25,
            max_new_tokens=200,
            min_length=0,
            attack_type='single-single'
        ),
        'k-t': lambda: PercentageCopyPasteAttack(
            tokenizer=tokenizer,
            num_insertions=3,
            insertion_ratio=0.25,
            max_new_tokens=200,
            min_length=0,
            attack_type='k-t'
        ),
        'Translation': lambda: BackTranslationTextEditor(
            translate_to_intermediary=Translator(from_lang="en", to_lang="zh").translate,
            translate_to_source=Translator(from_lang="zh", to_lang="en").translate
        )
    }
    
    if params.attack_name in attack_map:
        return attack_map[params.attack_name]()
    else:
        raise ValueError(f"Unsupported attack type: {params.attack_name}")

def assess_detection(params: ExperimentParams):
    """Basic detection experiment"""
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # Create dataset
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # Create watermark
    watermark = create_watermark(params, transformers_config)
    
    # Create pipelines
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        watermarked_texts_path=params.watermarked_texts_path,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=params.generation_mode,
        use_winmax=params.use_winmax
    )
    
    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode=params.text_source_mode,
        use_winmax=params.use_winmax
    )
    
    # Execute evaluation
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )
    
    print(f"metrics: {metrics}")
    return metrics

def assess_robustness_v2(params: ExperimentParams):
    """Robustness test experiment"""
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # Create dataset
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # Create watermark
    watermark = create_watermark(params, transformers_config)
    
    # Create attack
    attack = create_attack(params, tokenizer, transformers_config)
    attack_list = [attack] if attack else []
    
    # Create pipelines
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        watermarked_texts_path=params.watermarked_texts_path,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=params.generation_mode,
        text_editor_list=attack_list,
        use_winmax=params.use_winmax
    )
    
    # If attack is k-t or single-single, don't apply attack to unwm
    if params.attack_name == 'k-t' or params.attack_name == 'single-single':
        attack_list = []

    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode=params.text_source_mode,
        text_editor_list=attack_list,
        use_winmax=params.use_winmax
    )
    
    # Execute evaluation
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )
    
    print(f"metrics: {metrics}")
    return metrics

def assess_signature_detection(params: ExperimentParams):
    """Signature detection experiment"""
    if params.is_exp_algorithm:
        raise ValueError("EXP algorithm does not support signature detection")
    
    if params.n is None:
        raise ValueError("Signature detection requires n parameter")
        
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # Create dataset
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # Create watermark
    watermark = create_watermark(params, transformers_config)
    
    # Signature configuration
    signature_config = {
        'use_ngram': True,
        'n': params.n,
    }
    
    # Create pipelines
    wm_signature_pipeline = SignatureAwareWatermarkDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        watermarked_texts_path=params.watermarked_texts_path,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=params.generation_mode,
        signature_config=signature_config,
        use_winmax=params.use_winmax
    )

    unwm_signature_pipeline = SignatureAwareUnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=params.output_dir,
        extract_colors=params.extract_colors,  
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,  
        text_source_mode=params.text_source_mode,
        signature_config=signature_config,
        use_winmax=params.use_winmax
    )
    
    # Execute evaluation
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    signature_metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {signature_metrics}")

    print("========= no signature detection ==========")
    # Standard detection comparison
    standard_metrics = assess_detection(params)
    print(f"standard metrics: {standard_metrics}")
    
    return signature_metrics, standard_metrics

def assess_signature_robustness(params: ExperimentParams):
    """Signature robustness experiment"""
    if params.is_exp_algorithm:
        raise ValueError("EXP algorithm does not support signature detection")
    
    if params.n is None:
        raise ValueError("Signature robustness test requires n parameter")
        
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # Create dataset
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # Create watermark
    watermark = create_watermark(params, transformers_config)
    
    # Create attack
    attack = create_attack(params, tokenizer, transformers_config)
    attack_list = [attack] if attack else []
    
    # Signature configuration
    signature_config = {
        'use_ngram': True,
        'n': params.n,
    }
    
    # Create pipelines
    wm_signature_pipeline = SignatureAwareWatermarkDetectionPipeline_V2(
        dataset=dataset,
        text_editor_list=attack_list,
        watermark=watermark,
        output_dir=params.output_dir,
        watermarked_texts_path=params.watermarked_texts_path,
        extract_colors=params.extract_colors,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=params.generation_mode,
        signature_config=signature_config,
        use_winmax=params.use_winmax
    )

    unwm_signature_pipeline = SignatureAwareUnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        text_editor_list=attack_list,
        watermark=watermark,
        output_dir=params.output_dir,
        extract_colors=params.extract_colors,  
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode=params.text_source_mode,
        signature_config=signature_config,
        use_winmax=params.use_winmax
    )
    
    # Execute evaluation
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    signature_metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {signature_metrics}")

    print("========= no signature detection ==========")
    # Standard robustness test comparison
    standard_metrics = assess_robustness_v2(params)
    print(f"standard metrics: {standard_metrics}")
    
    return signature_metrics, standard_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--attack', type=str, default=None, choices=['Word-D', 'Word-S', 'Word-S-Context', 'scramble', 'single-single', 'k-t'])
    parser.add_argument('--dataset', type=str, default='dataset/c4/processed_c4.json', help='dataset path')
    parser.add_argument('--max_samples', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='data_1000/llama3.1/kgw/enc4_d1')
    parser.add_argument('--watermarked_texts_path', type=str, default='data_1000/llama3.1/kgw/enc4_d1/watermarked_texts.json')
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--generation_mode', type=str, default='load')
    parser.add_argument('--n', type=int, default=None, help='N-gram value for signature config')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--use_winmax', action='store_true', help='use winmax detection for kgw watermark')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', 
                        help='model name',
                        choices=[
                            'meta-llama/Llama-3.1-8B-Instruct',
                            'meta-llama/Llama-3.1-8B', 
                            'facebook/opt-1.3b',
                        ])
    args = parser.parse_args()

    params = ExperimentParams(
        algorithm_name=args.algorithm,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        watermarked_texts_path=args.watermarked_texts_path,
        dataset_path=args.dataset,
        delta=args.delta,
        temperature=args.temperature,
        generation_mode=args.generation_mode,
        attack_name=args.attack,
        use_winmax=args.use_winmax,
        n=args.n,
        model_name=args.model_name
    )
    
    print(f"Experiment configuration: \n {params}")

    if not params.attack_name and not params.n:
        assess_detection(params)
    elif params.attack_name and not params.n:
        assess_robustness_v2(params)
    elif not params.attack_name and params.n:
        assess_signature_detection(params)
    else:
        assess_signature_robustness(params)