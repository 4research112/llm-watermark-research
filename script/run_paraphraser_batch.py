import functools
import os
import subprocess
import shutil

def call_counter(func):
    @functools.wraps(func)
    def helper(*args, **kwargs):
        helper.calls += 1
        print(helper.calls)
        return func(*args, **kwargs)
    helper.calls = 0
    return helper

@call_counter
def run_command(cmd, output_file):
    print(f"Command: {cmd}")
    print(f"Output: {output_file}")
    print("-" * 80)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        subprocess.run(cmd, shell=True, stdout=f)
    print(f"Done: {output_file}")

def generate():
    #model, model_name = 'llama3.1', 'meta-llama/Llama-3.1-8B-Instruct'
    model, model_name = 'opt1.3', 'facebook/opt-1.3b'
    algorithms = ["KGW"]
    #algorithms = ["EXP"]
    max_samples = 10000
    
    datasets = [
        ("dataset/c4/processed_c4.json", "enc4")
        #('dataset/mbpp/mbpp.jsonl', 'mbpp')
    ]
    
    deltas = [0.5]
    
    for algorithm in algorithms:
      for dataset_path, dataset_name in datasets:
        for delta in deltas:
          algorithm_lower = algorithm.lower()
          output_dir = f"data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}"
          watermarked_texts_path = f"{output_dir}/watermarked_texts.json"
          
          cmd = (f"python3 script/paraphraser1.py "
                 f"--algorithm {algorithm} "
                 f"--max_samples {max_samples} "
                 f"--model_name {model_name} "
                 f"--output_dir {output_dir} "
                 f"--watermarked_texts_path {watermarked_texts_path} "
                 f"--dataset {dataset_path} "
                 f"--generation_mode generate")
          
          if algorithm == "EXP":
            cmd += f" --temperature {delta}"
          else:
            cmd += f" --delta {delta}" 

          output_file = f"{output_dir}/res.txt"
          run_command(cmd, output_file)

def detect():
    # model = "llama3.1"
    model = 'opt1.3'
    algorithm = "Unigram"
    max_samples = 5000
    
    datasets = [
        # ("dataset/zhtw/processed_zhtw_c4.json", "zhc4"),
        ("dataset/c4/processed_c4.json", "enc4")
    ]
    
    deltas = [2, 1, 0.8]
    # n_grams = [2]
    n_grams = [1, 3, 4, 5]
    
    for dataset_path, dataset_name in datasets:
        for delta in deltas:
            for n in n_grams:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/{n}-gram"
                watermarked_texts_path = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/1-gram/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser1.py "
                       f"--algorithm {algorithm} "
                       f"--max_samples {max_samples} "
                       f"--output_dir {output_dir} "
                       f"--watermarked_texts_path {watermarked_texts_path} "
                       f"--dataset {dataset_path} "
                       f"--generation_mode=load "
                       f"--delta={delta} "
                       f"--n={n}")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def code_generation():
    # model = "llama3.1"
    # algorithms = ["KGW", "SWEET", "Unigram"]
    # max_samples = 1000
    # model = 'opt1.3b'
    model = 'llama3.1'
    algorithms = ["EXP"]
    max_samples = 1
    
    datasets = [
        # ("dataset/human_eval/test.jsonl", "he")
        # ("dataset/mbpp/mbpp.jsonl", "mbpp")
        ('dataset/c4/processed_c4.json', 'enc4')
    ]
    
    deltas = [1, 0.8, 0.5]

    
    for dataset_path, dataset_name in datasets:
        for algorithm in algorithms:
            for delta in deltas:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}"
                watermarked_texts_path = f"{output_dir}/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser1.py "
                        f"--algorithm {algorithm} "
                        f"--max_samples {max_samples} "
                        f"--output_dir {output_dir} "
                        f"--watermarked_texts_path {watermarked_texts_path} "
                        f"--dataset {dataset_path} "
                        f"--generation_mode=generate "
                        f"--delta={delta} ")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def attack():
    # make sure use the robustness function in paraphraser1.py
    
    model, model_name = 'llama3.1', 'meta-llama/Llama-3.1-8B-Instruct'
    max_samples = 1000
    algorithms = ["SWEET"]
    #algorithms = ["EXP"]
    attack_names = ['scramble', 'Word-D', 'Word-S', 'Word-S-Context', 'single-single']
    #attack_names = ['single-single']
    #attack_names = ['scramble']

    datasets = [
        ('dataset/c4/processed_c4.json', 'enc4'),
        #('dataset/mbpp/mbpp.jsonl', 'mbpp')
    ]

    # deltas = [1, 0.8, 0.5]
    deltas = [1]

    G_USE_WINMAX = True

    winmax = 'winmax_' if G_USE_WINMAX else ''

    for dataset_path, dataset_name in datasets:
        for algorithm in algorithms:
            for delta in deltas:
                for attack_name in attack_names:
                    algorithm_lower = algorithm.lower()
                    output_dir = f"tables_data_{max_samples}_attack/{model}/{winmax}{algorithm_lower}/{dataset_name}_d{delta}/{attack_name.lower()}"
                    watermarked_texts_path = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/watermarked_texts.json"
                    

                    cmd = (f"python3 script/paraphraser1.py "
                        f"--algorithm {algorithm} "
                        f"--max_samples {max_samples} "
                        f"--output_dir {output_dir} "
                        f"--watermarked_texts_path {watermarked_texts_path} "
                        f"--dataset {dataset_path} "
                        f"--generation_mode load "
                        f"--model_name {model_name} "
                        f"--attack {attack_name}")

                    if algorithm == "EXP":
                      cmd += f" --temperature {delta}"
                    else:
                      cmd += f" --delta {delta}" 

                    if G_USE_WINMAX:
                        cmd += " --use_winmax"
                    
                    output_file = f"{output_dir}/res.txt"
                    run_command(cmd, output_file)


def main():
    # detect()
    # code_generation()
    generate()
    #attack()


if __name__ == "__main__":
    main() 
