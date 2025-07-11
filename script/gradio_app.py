import gradio as gr
import json
import sys
import io
import os
import contextlib
import subprocess
import time
import threading
from script.paraphraser1 import (
    assess_detection, 
    assess_robustness_v2, 
    assess_signature_detection, 
    assess_signature_robustness,
    ExperimentParams
)

def get_gpu_info():
    """獲取 GPU 使用情況 - 簡化版"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                used, total = map(int, lines[0].split(', '))
                current_time = time.strftime('%H:%M:%S')
                return f"[{current_time}] GPU: {used}MB / {total}MB ({used/total*100:.1f}%)"
            return "無法取得 GPU 資訊"
        else:
            return "nvidia-smi 執行失敗"
    except subprocess.TimeoutExpired:
        return "nvidia-smi 超時"
    except FileNotFoundError:
        return "未安裝 NVIDIA 驅動"
    except Exception as e:
        return f"錯誤: {str(e)}"

def update_gpu_info():
    """定時更新GPU資訊"""
    return get_gpu_info()

def generate_dynamic_output_dir(algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram):
    """動態生成輸出目錄"""
    # 自定義模型名稱簡化映射
    model_mapping = {
        'meta-llama/Llama-3.1-8B-Instruct': 'llama3.1',
        'meta-llama/Llama-3.1-8B': 'llama3.1',
        'facebook/opt-1.3b': 'opt1.3b',
        'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1': 'taide'
    }
    model_short = model_mapping.get(model_name, model_name.split('/')[-1].lower())
    
    # 提取數據集名稱並添加語言前綴
    dataset_name = os.path.basename(dataset_path).replace('processed_', '').replace('.json', '')
    if dataset_name == 'c4':
        dataset_name = 'enc4'  # c4 -> enc4
    
    # 算法名稱小寫
    algorithm_lower = algorithm.lower()
    
    # 根據算法選擇參數部分
    if algorithm.upper() == "EXP":
        param_part = f"t{temperature}"
    else:
        param_part = f"d{delta}"
    
    # 組合路徑（移除 gram 部分）
    output_dir = f"tables_data_{max_samples}/{model_short}/{algorithm_lower}/{dataset_name}_{param_part}"
    
    return output_dir

def generate_watermarked_texts_path(model_name, algorithm, dataset_path, delta, temperature):
    """動態生成水印文本路徑"""
    # 自定義模型名稱簡化映射
    model_mapping = {
        'meta-llama/Llama-3.1-8B-Instruct': 'llama3.1',
        'meta-llama/Llama-3.1-8B': 'llama3.1', 
        'facebook/opt-1.3b': 'opt1.3b',
        'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1': 'taide'
    }
    model_short = model_mapping.get(model_name, model_name.split('/')[-1].lower())
    
    # 提取數據集名稱並添加語言前綴（與輸出目錄邏輯一致）
    dataset_name = os.path.basename(dataset_path).replace('processed_', '').replace('.json', '')
    if dataset_name == 'c4':
        dataset_name = 'enc4'  # c4 -> enc4
    
    # 算法名稱小寫
    algorithm_lower = algorithm.lower()
    
    # 根據算法選擇參數部分
    if algorithm.upper() == "EXP":
        param_part = f"t{temperature}"
    else:
        param_part = f"d{delta}"
    
    # 組合水印文本路徑（使用動態數據集名稱）
    watermarked_path = f"texts1000/{model_short}/{algorithm_lower}/{dataset_name}_{param_part}/watermarked_texts.json"
    
    return watermarked_path

def generate_python_command(experiment_type, algorithm, model_name, dataset_path, max_samples, 
                          watermarked_texts_path, generation_mode, delta, temperature, 
                          attack_type, use_winmax, n_gram, output_dir):
    """生成對應的 Python 命令"""
    cmd_parts = ["python3", "script/paraphraser1.py"]
    
    # 基本參數
    cmd_parts.extend([
        "--algorithm", algorithm,
        "--dataset", dataset_path,
        "--max_samples", str(max_samples),
        "--output_dir", output_dir,
        "--model_name", model_name,
        "--generation_mode", generation_mode
    ])
    
    # 水印文本路徑
    if watermarked_texts_path.strip():
        cmd_parts.extend(["--watermarked_texts_path", watermarked_texts_path])
    
    # 算法特定參數
    if algorithm.upper() == "EXP":
        cmd_parts.extend(["--temperature", str(temperature)])
    else:
        cmd_parts.extend(["--delta", str(delta)])
    
    # 攻擊參數
    if experiment_type in ["watermark_robustness", "watermark_signature_robustness"] and attack_type:
        cmd_parts.extend(["--attack", attack_type])
    
    # WinMax 參數
    if use_winmax:
        cmd_parts.append("--use_winmax")
    
    # N-gram 參數
    if experiment_type in ["watermark_signature", "watermark_signature_robustness"]:
        cmd_parts.extend(["--n", str(n_gram)])
    
    return " ".join(cmd_parts)



def run_experiment(experiment_type, algorithm, model_name, dataset_path, max_samples, output_dir, 
                  watermarked_texts_path, generation_mode, delta, temperature, 
                  attack_type, use_winmax, n_gram, progress=gr.Progress()):
    
    progress(0, desc="初始化實驗...")
    
    # 使用動態生成的輸出目錄和水印文本路徑
    dynamic_output_dir = generate_dynamic_output_dir(
        algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram
    )
    
    dynamic_watermarked_path = generate_watermarked_texts_path(
        model_name, algorithm, dataset_path, delta, temperature
    )
    
    # 創建 ExperimentParams 對象
    params = ExperimentParams(
        algorithm_name=algorithm,
        max_samples=max_samples,
        output_dir=dynamic_output_dir,
        dataset_path=dataset_path,
        watermarked_texts_path=dynamic_watermarked_path,  # 使用動態生成的水印文本路徑
        generation_mode=generation_mode,
        delta=delta,
        temperature=temperature,
        attack_name=attack_type if experiment_type in ["watermark_robustness"] else None,
        use_winmax=use_winmax,
        n=n_gram,
        model_name=model_name
    )
    
    # 確保輸出目錄存在
    os.makedirs(dynamic_output_dir, exist_ok=True)
    res_file_path = os.path.join(dynamic_output_dir, "res.txt")
    
    # 捕獲輸出
    output_buffer = io.StringIO()
    
    try:
        progress(0.3, desc="執行實驗中...")
        
        with contextlib.redirect_stdout(output_buffer):
            if experiment_type == "watermark":
                assess_detection(params)
            elif experiment_type == "watermark_robustness":
                assess_robustness_v2(params)
            elif experiment_type == "watermark_signature":
                assess_signature_detection(params)                
            elif experiment_type == "watermark_signature_robustness":
                assess_signature_robustness(params)
        
        progress(1.0, desc="實驗完成!")
        
        output_text = output_buffer.getvalue()
        
        # 保存輸出到文件
        try:
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 水印實驗結果 ===\n")
                f.write(f"實驗類型: {experiment_type}\n")
                f.write(f"算法: {algorithm}\n")
                f.write(f"模型: {model_name}\n")
                f.write(f"使用 WinMax: {use_winmax}\n")
                f.write("=" * 50 + "\n\n")
                f.write(output_text)
            print(f"實驗結果已保存到: {res_file_path}")
        except Exception as save_error:
            print(f"保存文件失敗: {save_error}")
            
        output_text = f"實驗完成\n結果已保存到: {res_file_path}\n\n{output_text}"
        
        # 創建參數顯示字典
        params_dict = {
            'algorithm_name': params.algorithm_name,
            'model_name': params.model_name,
            'max_samples': params.max_samples,
            'output_dir': params.output_dir,
            'dataset_path': params.dataset_path,
            'watermarked_texts_path': params.watermarked_texts_path,
            'generation_mode': params.generation_mode,
            'delta': params.delta,
            'temperature': params.temperature,
            'attack_name': params.attack_name,
            'use_winmax': params.use_winmax,
            'n': params.n
        }
        
        return output_text, json.dumps(params_dict, indent=2, ensure_ascii=False)
        
    except Exception as e:
        progress(1.0, desc="實驗失敗")
        return f"實驗失敗: {str(e)}", ""

def update_interface_and_paths(experiment_type, algorithm, model_name, dataset_path, max_samples, 
                              watermarked_texts_path, generation_mode, delta, temperature, 
                              attack_type, use_winmax, n_gram):
    # 根據實驗類型顯示相關組件
    show_attack = experiment_type in ["watermark_robustness"]
    show_signature = False  # 移除簽名相關功能
    show_winmax = True  
    show_watermarked_path = True  # 總是顯示
    
    # 動態生成輸出目錄
    dynamic_output_dir = generate_dynamic_output_dir(
        algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram
    )
    
    # 動態生成水印文本路徑
    dynamic_watermarked_path = generate_watermarked_texts_path(
        model_name, algorithm, dataset_path, delta, temperature
    )
    
    # 生成 Python 命令
    python_cmd = generate_python_command(
        experiment_type, algorithm, model_name, dataset_path, max_samples,
        dynamic_watermarked_path, generation_mode, delta, temperature,
        attack_type, use_winmax, n_gram, dynamic_output_dir
    )
    
    return [
        gr.update(visible=show_attack),  # attack_type
        gr.update(visible=show_winmax),  # use_winmax
        gr.update(visible=show_signature),  # n_gram
        gr.update(visible=show_watermarked_path, value=dynamic_watermarked_path),  # watermarked_texts_path
        gr.update(visible=show_watermarked_path),   # generation_mode
        gr.update(value=dynamic_output_dir),  # output_dir
        gr.update(value=python_cmd)  # python_command
    ]

# 創建 Gradio 界面
with gr.Blocks(title="水印實驗系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Demo pages for watermark experiment")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 實驗設定")
            
            experiment_type = gr.Dropdown(
                choices=["watermark", "watermark_robustness"],
                label="實驗類型",
                value="watermark"
            )
            
            algorithm = gr.Dropdown(
                choices=["KGW", "SWEET", "Unigram", "EXP"],
                label="水印演算法",
                value="KGW"
            )
            
            model_name = gr.Dropdown(
                choices=[
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "facebook/opt-1.3b",
                    "meta-llama/Llama-3.1-8B",
                    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
                ],
                label="模型選擇",
                value="meta-llama/Llama-3.1-8B-Instruct"
            )
            
            dataset_path = gr.Dropdown(
                choices=[
                    "dataset/c4/processed_c4.json",
                    "dataset/zhtw/processed_zhtw_c4.json",
                    "dataset/human_eval/processed_human_eval.json",
                    "dataset/mbpp/processed_mbpp.json"
                ],
                label="資料集",
                value="dataset/c4/processed_c4.json"
            )
            
            max_samples = gr.Number(
                label="樣本數量",
                value=1,
                minimum=1,
                maximum=5000
            )
            
            output_dir = gr.Textbox(
                label="輸出目錄 (動態生成)",
                value="meeting/gradio",
                info="根據參數自動生成路徑"
            )
            
            # 動態顯示的組件
            watermarked_texts_path = gr.Textbox(
                label="水印文本路徑 (動態生成)",
                value="texts1000/llama3.1/kgw/enc4_d1.0/watermarked_texts.json",
                info="根據模型和算法參數自動生成"
            )
            
            generation_mode = gr.Dropdown(
                choices=["load", "generate"],
                label="生成模式",
                value="load"
            )
            
            with gr.Row():
                delta = gr.Number(
                    label="Delta for KGW-family",
                    value=1.0,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1
                )
                
                temperature = gr.Number(
                    label="Temperature for EXP",
                    value=1.0,
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1
                )
            
            # 條件顯示組件
            attack_type = gr.Dropdown(
                choices=["Word-D", "Word-S", "Word-S-Context", "scramble", 
                        "single-single", "k-t"],
                label="攻擊類型",
                value="Word-D",
                visible=False
            )
            
            use_winmax = gr.Checkbox(
                label="使用 WinMax 檢測",
                value=False,
                visible=True,
                info="增強型檢測模式，支援 KGW、SWEET、Unigram、EXP 水印算法"
            )
            
            n_gram = gr.Number(
                label="N-gram 值",
                value=2,
                minimum=1,
                maximum=5,
                visible=False
            )
            
            # Python 命令顯示
            python_command = gr.Textbox(
                label="Python 命令",
                lines=4,
                max_lines=6,
                value="python3 script/paraphraser1.py --algorithm KGW",
                info="可以複製到終端執行",
                show_copy_button=True
            )
            
            run_btn = gr.Button("執行實驗", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## 執行結果")
            
            output_text = gr.Textbox(
                label="執行輸出",
                lines=15,
                max_lines=20,
                show_copy_button=True
            )
            
            params_text = gr.Code(
                label="實驗參數",
                language="json"
            )

            # GPU 使用量顯示
            with gr.Row():
                gpu_info_text = gr.Textbox(
                    label="GPU 使用量 (自動更新)",
                    lines=1,
                    value=get_gpu_info(),
                    interactive=False,
                    scale=4
                )
                
                refresh_gpu_btn = gr.Button(
                    "🔄 刷新",
                    size="sm",
                    scale=1
                )
            
            # 隱藏的定時器組件
            timer_state = gr.State(value=0)
    
    # 所有可能影響輸出路徑和命令的組件
    update_inputs = [
        experiment_type, algorithm, model_name, dataset_path, max_samples,
        watermarked_texts_path, generation_mode, delta, temperature,
        attack_type, use_winmax, n_gram
    ]
    
    update_outputs = [
        attack_type, use_winmax, n_gram, watermarked_texts_path, 
        generation_mode, output_dir, python_command
    ]
    
    # 事件處理 - 動態更新
    for component in [experiment_type, algorithm, model_name, dataset_path, max_samples, 
                     delta, temperature, attack_type, use_winmax, n_gram]:
        component.change(
            fn=update_interface_and_paths,
            inputs=update_inputs,
            outputs=update_outputs
        )
    
    # 頁面載入時初始化
    demo.load(
        fn=update_interface_and_paths,
        inputs=update_inputs,
        outputs=update_outputs
    )
    
    # GPU 資訊刷新按鈕
    refresh_gpu_btn.click(
        fn=update_gpu_info,
        outputs=gpu_info_text
    )
    
    # 使用更現代的方法實現自動刷新
    if hasattr(gr, 'Timer'):
        # 如果Gradio支持Timer組件
        timer = gr.Timer(value=3)  # 每3秒觸發
        timer.tick(
            fn=update_gpu_info,
            outputs=gpu_info_text
        )
    else:
        # 後備方案：使用JavaScript定時器
        demo.load(
            fn=None,
            js="""
            function() {
                console.log('Setting up GPU info auto-refresh...');
                setInterval(function() {
                    // 查找刷新按鈕
                    const buttons = document.querySelectorAll('button');
                    for (let btn of buttons) {
                        if (btn.textContent.includes('🔄') || btn.textContent.includes('刷新')) {
                            btn.click();
                            break;
                        }
                    }
                }, 3000);
                return [];
            }
            """
        )
    
    run_btn.click(
        fn=run_experiment,
        inputs=[
            experiment_type, algorithm, model_name, dataset_path, max_samples, output_dir,
            watermarked_texts_path, generation_mode, delta, temperature,
            attack_type, use_winmax, n_gram
        ],
        outputs=[output_text, params_text]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 