import argparse
from utils.model_loader import load_model_and_config
from evaluation.tools.text_editor import TaideParaphraser
from translate import Translator
from evaluation.dataset import C4Dataset, ZHTWC4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType, WMTextDetectionPipeline, WatermarkedTextDetectionPipeline_V2, UnwatermarkedTextDetectionPipeline_V2, SignatureAwareWatermarkDetectionPipeline_V2, SignatureAwareUnwatermarkedTextDetectionPipeline_V2
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor
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

def get_transformes_config():
    # Transformers config
    model_name = 'facebook/opt-1.3b'
    # model_name = 'meta-llama/Llama-3.1-8B'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    # model_name = 'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1'
    print(f"使用模型: {model_name}")

    if model_name == 'facebook/opt-1.3b':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        transformers_config = TransformersConfig(
                model=model,
                tokenizer=tokenizer,
                vocab_size=50272,
                device=device,
                max_new_tokens=200,
                min_length=230,
                no_repeat_ngram_size=4,
                do_sample=True,
                eos_token_id=None,
            )
    else:    
        # Config for loading the model with quantization
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

        transformers_config = TransformersConfig(
                model=model,
                tokenizer=tokenizer,
                vocab_size=len(list(tokenizer.get_vocab().values())),
                device=device,
                max_new_tokens=200,
                min_length=230,
                no_repeat_ngram_size=4,
                do_sample=True,
                eos_token_id=None,
            )
    return model, tokenizer, transformers_config

def test_taide_paraphraser():
    # 載入模型、tokenizer 和配置
    model, tokenizer, transformers_config = load_model_and_config()
    
    # 創建 TaideParaphraser 實例
    paraphraser = TaideParaphraser(
        tokenizer=tokenizer,
        model=model,
        transformers_config=transformers_config,
        prompt="請重寫以下文字，保持原意但使用不同表達方式："
    )
    
    # 測試文本
    original_text = "人工智能正在改變我們的生活方式，從日常任務到複雜的科學研究都有其應用。"
    
    # 使用 paraphraser 改寫文本
    paraphrased_text = paraphraser.edit(original_text)
    
    print("原始文本:", original_text)
    print("改寫文本:", paraphrased_text)
    
    return paraphrased_text

def assess_robustness(algorithm_name, attack_name):
    # 加載模型和配置
    model, tokenizer, transformers_config = load_model_and_config()

    # my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    my_dataset = ZHTWC4Dataset('dataset/zhtw/processed_zhtw_c4.json', tokenizer=tokenizer, max_samples=3) 
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
    
    if attack_name == 'Doc-P(Taide)':
        attack = TaideParaphraser(
            tokenizer=tokenizer,
            model=model,
            transformers_config=transformers_config,
            prompt="請重寫以下文字，保持原意但使用不同表達方式："
        )

    return_type = DetectionPipelineReturnType.IS_WATERMARKED
    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[attack],
                                                show_progress=True, return_type=return_type) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=return_type)

    # calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    calculator = FundamentalSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'])
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))

def assess_robustness_1(algorithm_name, attack_name, max_samples, output_dir, watermarked_texts_path):
    # 加載模型和配置
    # model, tokenizer, transformers_config = load_model_and_config()
    model, tokenizer, transformers_config = get_transformes_config()

    my_dataset = ZHTWC4Dataset('dataset/zhtw/processed_zhtw_c4.json', tokenizer=tokenizer, max_samples=max_samples) 
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
    
    if attack_name == 'Doc-P(Taide)':
        attack = TaideParaphraser(
            tokenizer=tokenizer,
            model=model,
            transformers_config=transformers_config,
            prompt="請用繁體中文重寫以下文字，保持原意但使用不同表達方式："
        )
    elif attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.3)
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)

    return_type = DetectionPipelineReturnType.IS_WATERMARKED
    pipline1 = WMTextDetectionPipeline(dataset=my_dataset, text_editor_list=[attack],
                                                show_progress=True, return_type=return_type, watermarked_texts_path=watermarked_texts_path, output_dir=output_dir) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=return_type)

    # calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    calculator = FundamentalSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'])
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))

def assess_detection(algorithm_name: str, max_samples: int, output_dir: str, watermarked_texts_path: str, dataset_path: str, delta: float, generation_mode: str):
    # 加載模型和配置
    model, tokenizer, transformers_config = get_transformes_config()
    
    # 準備數據集和水印 
    if 'zhtw' in dataset_path.lower():
        print(f"使用 ZHTWC4Dataset 加載 {dataset_path}")
        dataset = ZHTWC4Dataset(dataset_path, tokenizer=tokenizer, max_samples=max_samples)
    elif 'c4' in dataset_path.lower():
        print(f"使用 C4Dataset 加載 {dataset_path}")
        dataset = C4Dataset(dataset_path, max_samples=max_samples)
    
    print(f"初始化 {algorithm_name} 水印...")
    print(f"delta: {args.delta}")
    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config,
        delta=args.delta
    )
    print(f"{algorithm_name} 水印初始化完成，實際 delta={watermark.config.delta}")
    
    # 初始化兩個 pipeline
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=generation_mode
    )
    
    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode='natural'  # 或 'generated'
    )
    
    # 執行評估並計算指標
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )
    
    print(f"metrics: {metrics}")

def assess_robustness_v2(algorithm_name: str, attack_name: str, max_samples: int, output_dir: str, watermarked_texts_path: str, dataset_path: str, delta: float, generation_mode: str):
    # 加載模型和配置
    model, tokenizer, transformers_config = get_transformes_config()
    
    # 準備數據集和水印 
    if 'zhtw' in dataset_path.lower():
        print(f"使用 ZHTWC4Dataset 加載 {dataset_path}")
        dataset = ZHTWC4Dataset(dataset_path, tokenizer=tokenizer, max_samples=max_samples)
    elif 'c4' in dataset_path.lower():
        print(f"使用 C4Dataset 加載 {dataset_path}")
        dataset = C4Dataset(dataset_path, max_samples=max_samples)
    
    print(f"初始化 {algorithm_name} 水印...")
    print(f"delta: {args.delta}")
    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config,
        delta=args.delta
    )
    print(f"{algorithm_name} 水印初始化完成，實際 delta={watermark.config.delta}")

    if attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.3)
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)
    
    # 初始化兩個 pipeline
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=generation_mode,
        text_editor_list=[attack]
    )
    
    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode='natural'  # 或 'generated'
    )
    
    # 執行評估並計算指標
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )
    
    print(f"metrics: {metrics}")

def assess_signature_detection(algorithm_name: str, max_samples: int, output_dir: str, watermarked_texts_path: str, dataset_path: str, delta: float, generation_mode: str, n: int):
    # 加載模型和配置
    model, tokenizer, transformers_config = get_transformes_config()
    
    # 準備數據集和水印 
    if 'zhtw' in dataset_path.lower():
        print(f"使用 ZHTWC4Dataset 加載 {dataset_path}")
        dataset = ZHTWC4Dataset(dataset_path, tokenizer=tokenizer, max_samples=max_samples)
    elif 'c4' in dataset_path.lower():
        print(f"使用 C4Dataset 加載 {dataset_path}")
        dataset = C4Dataset(dataset_path, max_samples=max_samples)
    
    print(f"初始化 {algorithm_name} 水印...")
    print(f"delta: {args.delta}")
    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config,
        delta=args.delta
    )
    print(f"{algorithm_name} 水印初始化完成，實際 delta={watermark.config.delta}")

    signature_config={
            'use_ngram': True,  # 是否使用 n-gram
            'n': n,  # n-gram 的 n 值
        }
    
    # 初始化兩個 pipeline
    wm_signature_pipeline = SignatureAwareWatermarkDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode=generation_mode,
        # signature_config=None,
        signature_config=signature_config
    )

    unwm_signature_pipeline = SignatureAwareUnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,  
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,  
        text_source_mode="natural",  
        # signature_config=None,
        signature_config=signature_config
    )
    
    # 執行評估並計算指標
    calculator = FundamentalSuccessRateCalculator(
        labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    )
    
    metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {metrics}")

    print("========= 沒有 signature 的偵測 ==========")
    # 沒有 signature 的偵測
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        generation_mode='load'
    )
    
    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,
        return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        text_source_mode='natural'  # 或 'generated'
    )

    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )

    print(f"standard metrics: {metrics}")

def assess_signature_robustness(algorithm_name: str, attack_name: str, max_samples: int, output_dir: str, watermarked_texts_path: str, dataset_path: str, delta: float, generation_mode: str, n: int):
    # 加載模型和配置
    model, tokenizer, transformers_config = get_transformes_config()
    
    # 準備數據集和水印 
    if 'zhtw' in dataset_path.lower():
        print(f"使用 ZHTWC4Dataset 加載 {dataset_path}")
        dataset = ZHTWC4Dataset(dataset_path, tokenizer=tokenizer, max_samples=max_samples)
    elif 'c4' in dataset_path.lower():
        print(f"使用 C4Dataset 加載 {dataset_path}")
        dataset = C4Dataset(dataset_path, max_samples=max_samples)
    
    print(f"初始化 {algorithm_name} 水印...")
    print(f"delta: {args.delta}")
    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config,
        delta=args.delta
    )
    print(f"{algorithm_name} 水印初始化完成，實際 delta={watermark.config.delta}")
    print(f"attack: {attack_name}")

    if attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.3)
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)
    elif attack_name == 'Word-S-Context':
        attack = ContextAwareSynonymSubstitution(ratio=0.5,
                                                 tokenizer=BertTokenizer.from_pretrained('bert-large-uncased', local_files_only=True),
                                                 model=BertForMaskedLM.from_pretrained('bert-large-uncased', local_files_only=True).to(device))
    elif attack_name == 'Translation':
        attack = BackTranslationTextEditor(translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                                           translate_to_source = Translator(from_lang="zh", to_lang="en").translate)
    elif attack_name == 'Doc-P-Taide':
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
        attack = TaideParaphraser(
            tokenizer=tokenizer,
            model=model,
            transformers_config=transformers_config,
            prompt='Please rewrite the following text: '
        )
    
    signature_config={
            'use_ngram': True,  # 是否使用 n-gram
            'n': n,  # n-gram 的 n 值
        }
    
    # 初始化兩個 pipeline
    wm_signature_pipeline = SignatureAwareWatermarkDetectionPipeline_V2(
        dataset=dataset,
        text_editor_list=[attack],
        watermark=watermark,
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        # return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        return_type=DetectionPipelineReturnType.SCORES,
        generation_mode=generation_mode,
        # signature_config=None,
        signature_config=signature_config
    )

    unwm_signature_pipeline = SignatureAwareUnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        text_editor_list=[attack],
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,  
        # return_type=DetectionPipelineReturnType.IS_WATERMARKED,  
        return_type=DetectionPipelineReturnType.SCORES,
        text_source_mode="natural",  
        # signature_config=None,
        signature_config=signature_config
    )
    
    # 執行評估並計算指標
    # calculator = FundamentalSuccessRateCalculator(
    #     labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']
    # )

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    
    metrics = calculator.calculate(
        wm_signature_pipeline.evaluate(),
        unwm_signature_pipeline.evaluate()
    )
    
    print(f"signature metrics: {metrics}")

    print("========= 沒有 signature 的偵測 ==========")
    # 沒有 signature 的偵測
    wm_pipeline = WatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        watermark=watermark,
        text_editor_list=[attack],
        output_dir=output_dir,
        watermarked_texts_path=watermarked_texts_path,
        extract_colors=True,
        # return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        return_type=DetectionPipelineReturnType.SCORES,
        generation_mode='load'
    )
    
    unwm_pipeline = UnwatermarkedTextDetectionPipeline_V2(
        dataset=dataset,
        text_editor_list=[attack],
        watermark=watermark,
        output_dir=output_dir,
        extract_colors=True,
        # return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        return_type=DetectionPipelineReturnType.SCORES,
        text_source_mode='natural'  # 或 'generated'
    )

    metrics = calculator.calculate(
        wm_pipeline.evaluate(),
        unwm_pipeline.evaluate()
    )

    print(f"standard metrics: {metrics}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Unigram')
    parser.add_argument('--attack', type=str, default='Word-D')
    parser.add_argument('--dataset', type=str, default='dataset/c4/processed_c4.json',
                        help='數據集路徑')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='meeting')
    parser.add_argument('--watermarked_texts_path', type=str, default='texts1000/llama3.1/unigram/enc4_d2/watermarked_texts.json')
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--generation_mode', type=str, default='load')
    parser.add_argument('--n', type=int, default=2, help='N-gram value for signature config')
    args = parser.parse_args()

    # assess_robustness(args.algorithm, args.attack)
    # assess_robustness_1(args.algorithm, args.attack, args.max_samples, args.output_dir, args.watermarked_texts_path)
    # test_taide_paraphraser()
    # 沒有 signature 的偵測
    # assess_detection(args.algorithm, args.max_samples, args.output_dir, args.watermarked_texts_path, args.dataset, args.delta, args.generation_mode)
    
    # 沒有 signature, 有 attack 的偵測 
    # assess_robustness_v2(args.algorithm, args.attack, args.max_samples, args.output_dir, args.watermarked_texts_path, args.dataset, args.delta, args.generation_mode)
    
    # 有 signature 的偵測
    assess_signature_detection(args.algorithm, args.max_samples, args.output_dir, args.watermarked_texts_path, args.dataset, args.delta, args.generation_mode, args.n)

    # 有 signature, 有 attack 的偵測
    # assess_signature_robustness(args.algorithm, args.attack, args.max_samples, args.output_dir, args.watermarked_texts_path, args.dataset, args.delta, args.generation_mode, args.n)