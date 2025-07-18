import argparse
from dataclasses import dataclass, field
from typing import Optional, List
from utils.model_loader import load_model_and_config
from evaluation.tools.text_editor import TaideParaphraser
from translate import Translator
from evaluation.dataset import C4Dataset, ZHTWC4Dataset, HumanEvalDataset, MBPPDataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType, WMTextDetectionPipeline, WatermarkedTextDetectionPipeline_V2, UnwatermarkedTextDetectionPipeline_V2, SignatureAwareWatermarkDetectionPipeline_V2, SignatureAwareUnwatermarkedTextDetectionPipeline_V2
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor, ScrambleAttack, CopyPasteAttack, PercentageCopyPasteAttack
import torch
import numpy as np
import random
from utils.timer import timer
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
    """實驗參數配置"""
    # 基本參數
    algorithm_name: str
    max_samples: int
    output_dir: str
    dataset_path: str
    # 模型選擇
    model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'
    # 水印參數
    delta: float = 1.0
    temperature: float = 1.0
    # 檔案路徑
    watermarked_texts_path: Optional[str] = None
    # 模式設定
    generation_mode: str = "load"
    # 攻擊參數
    attack_name: Optional[str] = None
    use_winmax: bool = False
    # n-gram signature config
    n: Optional[int] = None
    # 其他參數
    extract_colors: bool = True
    text_source_mode: str = "natural"
    
    def __post_init__(self):
        """參數驗證"""
        if self.algorithm_name not in ['KGW', 'SWEET', 'Unigram', 'EXP']:
            raise ValueError(f"不支援的算法: {self.algorithm_name}")
        
        if self.generation_mode not in ['load', 'generate']:
            raise ValueError(f"不支援的生成模式: {self.generation_mode}")
            
        # 驗證模型名稱
        supported_models = [
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3.1-8B',
            'facebook/opt-1.3b',
            'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1'
        ]
        if self.model_name not in supported_models:
            print(f"警告: 模型 {self.model_name} 可能不被支援，支援的模型: {supported_models}")
    
    @property
    def is_exp_algorithm(self) -> bool:
        """是否為 EXP 算法"""
        return self.algorithm_name == 'EXP'
    
    def get_watermark_param(self) -> dict:
        """獲取水印參數"""
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
    """數據集配置"""
    path: str
    tokenizer: object
    max_samples: int
    
    def create_dataset(self):
        """根據路徑創建對應的數據集"""
        if 'zhtw' in self.path.lower():
            print(f"使用 ZHTWC4Dataset 加載 {self.path}")
            return ZHTWC4Dataset(self.path, tokenizer=self.tokenizer, max_samples=self.max_samples)
        elif 'c4' in self.path.lower():
            print(f"使用 C4Dataset 加載 {self.path}")
            return C4Dataset(self.path, max_samples=self.max_samples)
        elif 'human_eval' in self.path.lower():
            print(f"使用 HumanEvalDataset 加載 {self.path}")
            return HumanEvalDataset(self.path, max_samples=self.max_samples)
        elif 'mbpp' in self.path.lower():
            print(f"使用 MBPPDataset 加載 {self.path}")
            return MBPPDataset(self.path, max_samples=self.max_samples)
        else:
            raise ValueError(f"不支援的數據集類型: {self.path}")

def get_transformes_config(model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'):
    """獲取 transformers 配置"""
    print(f"使用模型: {model_name}")

    # OPT 模型配置
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
        
    # Llama 和 TAIDE 模型配置
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
    """創建水印實例"""
    print(f"初始化 {params.algorithm_name} 水印...")
    
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
    
    print(f"{params.algorithm_name} 水印初始化完成")
    return watermark

def create_attack(params: ExperimentParams, tokenizer, transformers_config):
    """創建攻擊實例"""
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
        ),
        'Doc-P-Taide': lambda: _create_taide_attack(transformers_config)
    }
    
    if params.attack_name in attack_map:
        return attack_map[params.attack_name]()
    else:
        raise ValueError(f"不支援的攻擊類型: {params.attack_name}")

def _create_taide_attack(transformers_config):
    """創建 TAIDE 改寫攻擊"""
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1',
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained('taide/Llama3-TAIDE-LX-8B-Chat-Alpha1', local_files_only=True)
    
    return TaideParaphraser(
        tokenizer=tokenizer,
        model=model,
        transformers_config=transformers_config,
        prompt='Please rewrite the following text: '
    )

def assess_detection(params: ExperimentParams):
    """基本檢測實驗"""
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # 創建數據集
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # 創建水印
    watermark = create_watermark(params, transformers_config)
    
    # 創建 pipelines
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
    
    # 執行評估
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
    """魯棒性測試實驗"""
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # 創建數據集
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # 創建水印
    watermark = create_watermark(params, transformers_config)
    
    # 創建攻擊
    attack = create_attack(params, tokenizer, transformers_config)
    attack_list = [attack] if attack else []
    
    # 創建 pipelines
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
    
    # 如果攻擊是 k-t 或 single-single，則不對 unwm 使用攻擊
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
    
    # 執行評估
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
    """簽名檢測實驗"""
    if params.is_exp_algorithm:
        raise ValueError("EXP 算法不支援簽名檢測")
    
    if params.n is None:
        raise ValueError("簽名檢測需要指定 n 參數")
        
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # 創建數據集
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # 創建水印
    watermark = create_watermark(params, transformers_config)
    
    # 簽名配置
    signature_config = {
        'use_ngram': True,
        'n': params.n,
    }
    
    # 創建 pipelines
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
    
    # 執行評估
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    signature_metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {signature_metrics}")

    print("========= 沒有 signature 的偵測 ==========")
    # 標準檢測比較
    standard_metrics = assess_detection(params)
    print(f"standard metrics: {standard_metrics}")
    
    return signature_metrics, standard_metrics

def assess_signature_robustness(params: ExperimentParams):
    """簽名魯棒性實驗"""
    if params.is_exp_algorithm:
        raise ValueError("EXP 算法不支援簽名檢測")
    
    if params.n is None:
        raise ValueError("簽名魯棒性測試需要指定 n 參數")
        
    model, tokenizer, transformers_config = get_transformes_config(params.model_name)
    
    # 創建數據集
    dataset_config = DatasetConfig(params.dataset_path, tokenizer, params.max_samples)
    dataset = dataset_config.create_dataset()
    
    # 創建水印
    watermark = create_watermark(params, transformers_config)
    
    # 創建攻擊
    attack = create_attack(params, tokenizer, transformers_config)
    attack_list = [attack] if attack else []
    
    # 簽名配置
    signature_config = {
        'use_ngram': True,
        'n': params.n,
    }
    
    # 創建 pipelines
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
    
    # 執行評估
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    signature_metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {signature_metrics}")

    print("========= 沒有 signature 的偵測 ==========")
    # 標準魯棒性測試比較
    standard_metrics = assess_robustness_v2(params)
    print(f"standard metrics: {standard_metrics}")
    
    return signature_metrics, standard_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='SWEET')
    parser.add_argument('--attack', type=str, default=None, choices=['Word-D', 'Word-S', 'Word-S-Context', 'scramble', 'single-single', 'k-t'])
    parser.add_argument('--dataset', type=str, default='dataset/c4/processed_c4.json', help='數據集路徑')
    parser.add_argument('--max_samples', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='meeting/sweet_scramble_winmax')
    parser.add_argument('--watermarked_texts_path', type=str, default='texts1000/llama3.1/sweet/enc4_d1/watermarked_texts.json')
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--generation_mode', type=str, default='load')
    parser.add_argument('--n', type=int, default=None, help='N-gram value for signature config')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--use_winmax', action='store_true', help='use winmax detection for kgw watermark')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', 
                        help='模型名稱',
                        choices=[
                            'meta-llama/Llama-3.1-8B-Instruct',
                            'meta-llama/Llama-3.1-8B', 
                            'facebook/opt-1.3b',
                            'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1'
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
    
    print(f"實驗配置: \n {params}")

    if not params.attack_name and not params.n:
        assess_detection(params)
    elif params.attack_name and not params.n:
        assess_robustness_v2(params)
    elif not params.attack_name and params.n:
        assess_signature_detection(params)
    else:
        assess_signature_robustness(params)