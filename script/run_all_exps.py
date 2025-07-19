import functools
import os
import subprocess

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
    print(f"{cmd}")
    print(f"> {output_file}")
    print("-" * 80)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        subprocess.run(cmd, shell=True, stdout=f)
    print(f"Done: {output_file}")

def generate_all_texts():
    model = "llama3.1"
    algorithms = ["Unigram", "KGW", "SWEET", "EXP"]
    max_samples_list = [1000, 5000, 10000]
    
    datasets = [
        ("dataset/c4/processed_c4.json", "enc4")
    ]
    
    deltas = [2, 1, 0.8, 0.5]
    temperatures = [0.5, 0.3, 0.1]
    
    for dataset_path, dataset_name in datasets:
        for max_samples in max_samples_list:    
            for algorithm in algorithms:
                if algorithm == "EXP":
                    for temperature in temperatures:
                        algorithm_lower = algorithm.lower()
                        output_dir = f"data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_t{temperature}"
                    
                        cmd = (f"python3 script/paraphraser1.py "
                            f"--algorithm {algorithm} "
                            f"--max_samples {max_samples} "
                            f"--output_dir {output_dir} "
                            f"--dataset {dataset_path} "
                            f"--generation_mode=generate "
                            f"--temperature={temperature} "
                            )
                    
                        output_file = f"{output_dir}/res.txt"
                        run_command(cmd, output_file)
                else:
                    for delta in deltas:
                        algorithm_lower = algorithm.lower()
                        output_dir = f"data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}"
                    
                        cmd = (f"python3 script/paraphraser1.py "
                            f"--algorithm {algorithm} "
                            f"--max_samples {max_samples} "
                            f"--output_dir {output_dir} "
                            f"--dataset {dataset_path} "
                            f"--generation_mode=generate "
                            f"--delta={delta} "
                            )
                    
                        output_file = f"{output_dir}/res.txt"
                        run_command(cmd, output_file)

def main():
    generate_all_texts()

if __name__ == "__main__":
    main() 