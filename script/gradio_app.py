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
    """Get GPU usage - simplified version"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                used, total = map(int, lines[0].split(', '))
                current_time = time.strftime('%H:%M:%S')
                return f"[{current_time}] GPU: {used}MB / {total}MB ({used/total*100:.1f}%)"
            return "Failed to get GPU information"
        else:
            return "nvidia-smi execution failed"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timeout"
    except FileNotFoundError:
        return "NVIDIA driver not installed"
    except Exception as e:
        return f"Error: {str(e)}"

def update_gpu_info():
    """Update GPU information periodically"""
    return get_gpu_info()

def generate_dynamic_output_dir(algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram):
    """Generate dynamic output directory"""
    # Custom model name mapping
    model_mapping = {
        'meta-llama/Llama-3.1-8B-Instruct': 'llama3.1',
        'meta-llama/Llama-3.1-8B': 'llama3.1',
        'facebook/opt-1.3b': 'opt1.3b',
        'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1': 'taide'
    }
    model_short = model_mapping.get(model_name, model_name.split('/')[-1].lower())
    
    # Extract dataset name and add language prefix
    dataset_name = os.path.basename(dataset_path).replace('processed_', '').replace('.json', '')
    if dataset_name == 'c4':
        dataset_name = 'enc4'  # c4 -> enc4
    
    # Algorithm name lowercase
    algorithm_lower = algorithm.lower()
    
    # Select parameters based on algorithm
    if algorithm.upper() == "EXP":
        param_part = f"t{temperature}"
    else:
        param_part = f"d{delta}"
    
    # Combine path (remove gram part)
    output_dir = f"tables_data_{max_samples}/{model_short}/{algorithm_lower}/{dataset_name}_{param_part}"
    
    return output_dir

def generate_watermarked_texts_path(model_name, algorithm, dataset_path, delta, temperature):
    """Generate dynamic watermark text path"""
    # Custom model name mapping
    model_mapping = {
        'meta-llama/Llama-3.1-8B-Instruct': 'llama3.1',
        'meta-llama/Llama-3.1-8B': 'llama3.1', 
        'facebook/opt-1.3b': 'opt1.3b',
        'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1': 'taide'
    }
    model_short = model_mapping.get(model_name, model_name.split('/')[-1].lower())
    
    # Extract dataset name and add language prefix (consistent with output directory logic)
    dataset_name = os.path.basename(dataset_path).replace('processed_', '').replace('.json', '')
    if dataset_name == 'c4':
        dataset_name = 'enc4'  # c4 -> enc4
    
    # Algorithm name lowercase
    algorithm_lower = algorithm.lower()
    
    # Select parameters based on algorithm
    if algorithm.upper() == "EXP":
        param_part = f"t{temperature}"
    else:
        param_part = f"d{delta}"
    
    # Combine watermark text path (use dynamic dataset name)
    watermarked_path = f"texts1000/{model_short}/{algorithm_lower}/{dataset_name}_{param_part}/watermarked_texts.json"
    
    return watermarked_path

def generate_python_command(experiment_type, algorithm, model_name, dataset_path, max_samples, 
                          watermarked_texts_path, generation_mode, delta, temperature, 
                          attack_type, use_winmax, n_gram, output_dir):
    """Generate corresponding Python command"""
    cmd_parts = ["python3", "script/paraphraser1.py"]
    
    # Basic parameters
    cmd_parts.extend([
        "--algorithm", algorithm,
        "--dataset", dataset_path,
        "--max_samples", str(max_samples),
        "--output_dir", output_dir,
        "--model_name", model_name,
        "--generation_mode", generation_mode
    ])
    
    # Watermark text path
    if watermarked_texts_path.strip():
        cmd_parts.extend(["--watermarked_texts_path", watermarked_texts_path])
    
    # Algorithm specific parameters
    if algorithm.upper() == "EXP":
        cmd_parts.extend(["--temperature", str(temperature)])
    else:
        cmd_parts.extend(["--delta", str(delta)])
    
    # Attack parameters
    if experiment_type in ["watermark_robustness", "watermark_signature_robustness"] and attack_type:
        cmd_parts.extend(["--attack", attack_type])
    
    # WinMax parameters
    if use_winmax:
        cmd_parts.append("--use_winmax")
    
    # N-gram parameters
    if experiment_type in ["watermark_signature", "watermark_signature_robustness"]:
        cmd_parts.extend(["--n", str(n_gram)])
    
    return " ".join(cmd_parts)



def run_experiment(experiment_type, algorithm, model_name, dataset_path, max_samples, output_dir, 
                  watermarked_texts_path, generation_mode, delta, temperature, 
                  attack_type, use_winmax, n_gram, progress=gr.Progress()):
    
    progress(0, desc="Initializing experiment...")
    
    # Use dynamic generated output directory and watermark text path
    dynamic_output_dir = generate_dynamic_output_dir(
        algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram
    )
    
    dynamic_watermarked_path = generate_watermarked_texts_path(
        model_name, algorithm, dataset_path, delta, temperature
    )
    
    # Create ExperimentParams object
    params = ExperimentParams(
        algorithm_name=algorithm,
        max_samples=max_samples,
        output_dir=dynamic_output_dir,
        dataset_path=dataset_path,
        watermarked_texts_path=dynamic_watermarked_path,  # Use dynamic generated watermark text path
        generation_mode=generation_mode,
        delta=delta,
        temperature=temperature,
        attack_name=attack_type if experiment_type in ["watermark_robustness"] else None,
        use_winmax=use_winmax,
        n=n_gram,
        model_name=model_name
    )
    
    # Ensure output directory exists
    os.makedirs(dynamic_output_dir, exist_ok=True)
    res_file_path = os.path.join(dynamic_output_dir, "res.txt")
    
    # Capture output
    output_buffer = io.StringIO()
    
    try:
        progress(0.3, desc="Executing experiment...")
        
        with contextlib.redirect_stdout(output_buffer):
            if experiment_type == "watermark":
                assess_detection(params)
            elif experiment_type == "watermark_robustness":
                assess_robustness_v2(params)
            elif experiment_type == "watermark_signature":
                assess_signature_detection(params)                
            elif experiment_type == "watermark_signature_robustness":
                assess_signature_robustness(params)
        
        progress(1.0, desc="Experiment completed!")
        
        output_text = output_buffer.getvalue()
        
        # Save output to file
        try:
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Watermark experiment results ===\n")
                f.write(f"Experiment type: {experiment_type}\n")
                f.write(f"Algorithm: {algorithm}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"使用 WinMax: {use_winmax}\n")
                f.write("=" * 50 + "\n\n")
                f.write(output_text)
            print(f"Experiment results saved to: {res_file_path}")
        except Exception as save_error:
            print(f"Failed to save file: {save_error}")
            
        output_text = f"Experiment completed\nResults saved to: {res_file_path}\n\n{output_text}"
        
        # Create parameter display dictionary
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
        progress(1.0, desc="Experiment failed")
        return f"Experiment failed: {str(e)}", ""

def update_interface_and_paths(experiment_type, algorithm, model_name, dataset_path, max_samples, 
                              watermarked_texts_path, generation_mode, delta, temperature, 
                              attack_type, use_winmax, n_gram):
    # Show related components based on experiment type
    show_attack = experiment_type in ["watermark_robustness"]
    show_signature = False  # Remove signature-related features
    show_winmax = True  
    show_watermarked_path = True  # Always show
    
    # Generate dynamic output directory
    dynamic_output_dir = generate_dynamic_output_dir(
        algorithm, model_name, dataset_path, max_samples, delta, temperature, n_gram
    )
    
    # Generate dynamic watermark text path
    dynamic_watermarked_path = generate_watermarked_texts_path(
        model_name, algorithm, dataset_path, delta, temperature
    )
    
    # Generate Python command
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

# Create Gradio interface
with gr.Blocks(title="Watermark experiment system", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Demo pages for watermark experiment")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Experiment settings")
            
            experiment_type = gr.Dropdown(
                choices=["watermark", "watermark_robustness"],
                label="Experiment type",
                value="watermark"
            )
            
            algorithm = gr.Dropdown(
                choices=["KGW", "SWEET", "Unigram", "EXP"],
                label="Watermark algorithm",
                value="KGW"
            )
            
            model_name = gr.Dropdown(
                choices=[
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "facebook/opt-1.3b",
                    "meta-llama/Llama-3.1-8B",
                    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
                ],
                label="Model selection",
                value="meta-llama/Llama-3.1-8B-Instruct"
            )
            
            dataset_path = gr.Dropdown(
                choices=[
                    "dataset/c4/processed_c4.json",
                    "dataset/zhtw/processed_zhtw_c4.json",
                    "dataset/human_eval/processed_human_eval.json",
                    "dataset/mbpp/processed_mbpp.json"
                ],
                label="Dataset",
                value="dataset/c4/processed_c4.json"
            )
            
            max_samples = gr.Number(
                label="Sample number",
                value=1,
                minimum=1,
                maximum=5000
            )
            
            output_dir = gr.Textbox(
                label="Output directory (dynamic generation)",
                value="meeting/gradio",
                info="Path generated automatically based on parameters"
            )
            
            # Dynamic display components
            watermarked_texts_path = gr.Textbox(
                label="Watermark text path (dynamic generation)",
                value="texts1000/llama3.1/kgw/enc4_d1.0/watermarked_texts.json",
                info="Path generated automatically based on model and algorithm parameters"
            )
            
            generation_mode = gr.Dropdown(
                choices=["load", "generate"],
                label="Generation mode",
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
            
            # Conditional display components
            attack_type = gr.Dropdown(
                choices=["Word-D", "Word-S", "Word-S-Context", "scramble", 
                        "single-single", "k-t"],
                label="Attack type",
                value="Word-D",
                visible=False
            )
            
            use_winmax = gr.Checkbox(
                label="Use WinMax detection",
                value=False,
                visible=True,
                info="Enhanced detection mode, supports KGW, SWEET, Unigram, EXP watermark algorithms"
            )
            
            n_gram = gr.Number(
                label="N-gram value",
                value=2,
                minimum=1,
                maximum=5,
                visible=False
            )
            
            # Python command display
            python_command = gr.Textbox(
                label="Python command",
                lines=4,
                max_lines=6,
                value="python3 script/paraphraser1.py --algorithm KGW",
                info="Can be copied to terminal to execute",
                show_copy_button=True
            )
            
            run_btn = gr.Button("Run experiment", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## Execution results")
            
            output_text = gr.Textbox(
                label="Execution output",
                lines=15,
                max_lines=20,
                show_copy_button=True
            )
            
            params_text = gr.Code(
                label="Experiment parameters",
                language="json"
            )

            # GPU usage display
            with gr.Row():
                gpu_info_text = gr.Textbox(
                    label="GPU usage (auto-update)",
                    lines=1,
                    value=get_gpu_info(),
                    interactive=False,
                    scale=4
                )
                
                refresh_gpu_btn = gr.Button(
                    "🔄 Refresh",
                    size="sm",
                    scale=1
                )
            
            # Hidden timer component
            timer_state = gr.State(value=0)
    
    # All components that may affect output path and command
    update_inputs = [
        experiment_type, algorithm, model_name, dataset_path, max_samples,
        watermarked_texts_path, generation_mode, delta, temperature,
        attack_type, use_winmax, n_gram
    ]
    
    update_outputs = [
        attack_type, use_winmax, n_gram, watermarked_texts_path, 
        generation_mode, output_dir, python_command
    ]
    
    # Event handling - dynamic update
    for component in [experiment_type, algorithm, model_name, dataset_path, max_samples, 
                     delta, temperature, attack_type, use_winmax, n_gram]:
        component.change(
            fn=update_interface_and_paths,
            inputs=update_inputs,
            outputs=update_outputs
        )
    
    # Initialize when page loads
    demo.load(
        fn=update_interface_and_paths,
        inputs=update_inputs,
        outputs=update_outputs
    )
    
    # GPU info refresh button
    refresh_gpu_btn.click(
        fn=update_gpu_info,
        outputs=gpu_info_text
    )
    
    # Use more modern method to implement auto-refresh
    if hasattr(gr, 'Timer'):
        # If Gradio supports Timer component
        timer = gr.Timer(value=3)  # Trigger every 3 seconds
        timer.tick(
            fn=update_gpu_info,
            outputs=gpu_info_text
        )
    else:
        # Backup: use JavaScript timer
        demo.load(
            fn=None,
            js="""
            function() {
                console.log('Setting up GPU info auto-refresh...');
                setInterval(function() {
                    // Find refresh button
                    const buttons = document.querySelectorAll('button');
                    for (let btn of buttons) {
                        if (btn.textContent.includes('🔄') || btn.textContent.includes('Refresh')) {
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