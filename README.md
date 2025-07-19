# Signature Filtering for LLM Watermark Detection

<!-- Add a relevant meme or illustration here -->

How can we dramatically improve the detection accuracy of LLM watermarks without sacrificing text quality? Turns out the answer lies in strategically filtering specific token types during detection. Our signature filtering method achieves nearly perfect detection performance (TPR ≈ 1, FPR ≈ 0) even in the most challenging low-strength watermark settings.

This project implements signature filtering, a lightweight enhancement for KGW-style watermarking methods that optimally selects token types to amplify watermark signals during detection time.

<!-- ## How It Works -->

<!-- Explanation of the method -->

## Getting Started

### Prerequisites

- Python 3.10+
- Conda package manager

### Environment Setup

1. **Create and activate Conda virtual environment**

```bash
# Activate the myenv environment
conda activate myenv
```

2. **Install dependencies**

```bash
# If the environment doesn't exist yet, create it first and install dependencies
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

3. **Configure Python Path**

If you don't have `direnv` installed, manually set the PYTHONPATH to include the project root directory:

```bash
# Add the project root to PYTHONPATH (replace with your actual project path)
export PYTHONPATH="${PYTHONPATH}:/path/to/your/llm-watermark-signature"

# For the current session, you can also run:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# To make it permanent, add this line to your shell profile (.bashrc, .zshrc, etc.):
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/your/llm-watermark-signature"' >> ~/.bashrc
source ~/.bashrc
```

**Why this is needed:** This ensures Python can properly import modules from the project directory structure when running the scripts.


## Quick Start - Reproduce Experimental Results

### Run All Experiments

To reproduce all experimental results from the paper, simply execute:

```bash
python3 script/run_all_exps.py
```

This script will automatically run:
- **Four watermarking algorithms**: KGW, Unigram, SWEET, EXP
- **Multiple sample sizes**: 1,000, 5,000, 10,000 samples
- **Various parameter settings**: 
  - Delta values: 2.0, 1.0, 0.8, 0.5 (for KGW, SWEET, Unigram)
  - Temperature values: 0.5, 0.3, 0.1 (for EXP)
- **Automatic output management**: Results saved to `data_{max_samples}/{model}/{algorithm}/{dataset}_params/` directory structure

**Expected runtime**: Complete execution may take several hours to days depending on hardware configuration.

**Output location**: Each experiment's results will be saved in the corresponding directory's `res.txt` file.


## Evaluation Pipelines

### Single Experiment Execution

Run a single experiment configuration:

```bash
python3 script/paraphraser1.py \
    --algorithm KGW \
    --max_samples 1000 \
    --output_dir data_1000/llama3.1/kgw/enc4_d2 \
    --dataset dataset/c4/processed_c4.json \
    --generation_mode generate \
    --delta 2.0
```

**Key Parameters:**

- `--algorithm`: Watermarking algorithm (KGW, SWEET, Unigram, EXP)
- `--max_samples`: Number of experiment samples
- `--delta`: Watermark strength parameter for KGW-style watermark
- `--temperature`: Temperature parameter for EXP watermark
- `--dataset`: Dataset path
- `--generation_mode`: Mode selection (generate/load)
- `--use_winmax`: Enable WinMax detection mode for enhanced accuracy (applicable to all four algorithms)
- `--attack`: Attack type for robustness testing
- `--model_name`: Model selection (Llama-3.1, OPT-1.3b)

### Advanced Detection Features

**WinMax Detection Mode:**
Enhanced detection method that improves accuracy across all watermarking algorithms:

```bash
python3 script/paraphraser1.py \
    --algorithm KGW \
    --use_winmax \
    --max_samples 1000 \
    --output_dir data_1000/llama3.1/kgw_winmax/enc4_d2 \
    --dataset dataset/c4/processed_c4.json \
    --generation_mode generate \
    --delta 2.0
```

**Attack Robustness Testing:**
Test watermark resilience against various text modifications:

```bash
# Word deletion attack
python3 script/paraphraser1.py \
    --algorithm SWEET \
    --attack Word-D \
    --max_samples 1000 \
    --watermarked_texts_path data_1000/llama3.1/sweet/enc4_d1/watermarked_texts.json \
    --output_dir data_1000/llama3.1/sweet_attack/enc4_d1_word_d \
    --generation_mode load

# Scramble attack with WinMax
python3 script/paraphraser1.py \
    --algorithm KGW \
    --attack scramble \
    --use_winmax \
    --max_samples 1000 \
    --watermarked_texts_path data_1000/llama3.1/kgw/enc4_d2/watermarked_texts.json \
    --output_dir data_1000/llama3.1/kgw_attack/enc4_d2_scramble_winmax \
    --generation_mode load
```

**Supported Attack Types:**
- `Word-D`: Word deletion (removes 30% of words randomly)
- `Word-S`: Synonym substitution (replaces 50% of words with synonyms)
- `Word-S-Context`: Context-aware synonym substitution using BERT
- `scramble`: Token scrambling attack
- `single-single`: Copy-paste attack (single insertion)
- `k-t`: Multiple copy-paste insertions


### Batch Experiment Execution

Execute batch processing for multiple experiment configurations:

```bash
python3 script/run_paraphraser_batch.py
```

**Supported Experiment Types:**

- **Generation Experiments** (`generate()`): Generate and detect watermarked text
- **Detection Experiments** (`detect()`): Perform watermark detection on generated text (detection only)
- **Code Generation** (`code_generation()`): Watermarking experiments for code generation tasks
- **Attack Experiments** (`attack()`): Test watermark robustness under various attacks

<!-- ## Results -->

<!-- Experimental results tables -->

## Models and Datasets

**Models Used for Experiments:**
- meta-llama/Llama-3.1-8B-Instruct
- facebook/opt-1.3b

**Supported Datasets:**
- C4 Dataset
- MBPP (Mostly Basic Python Problems)

### Output Management

The experiments automatically organizes results using structured paths:
- Output directories: `data_{max_samples}/{model}/{algorithm}/{dataset}_{params}/`
- Watermark texts: `{output_dir}/watermarked_texts.json`
- Results saved to: `{output_dir}/res.txt`

<!-- ## Milestones -->

<!-- Project roadmap -->

<!-- ## Citation -->

<!-- Citation information -->

<!-- ## License -->

<!-- License information -->


## Web Interface (Gradio)

### Overview

This project includes a user-friendly web interface built with Gradio that provides an interactive way to run watermark experiments without command-line operations.

### Launch Gradio Interface

Start the web interface:

```bash
python gradio_app.py
```

The interface will be available at `http://localhost:7860`

### Features

**Experiment Configuration:**
- Select watermarking algorithms (KGW, SWEET, Unigram, EXP)
- Choose models (Llama-3.1, OPT-1.3b)
- Configure datasets (C4, MBPP)
- Adjust parameters (delta, temperature, sample count)

**Advanced Options:**
- WinMax detection mode for enhanced accuracy
- Attack simulation (Word deletion, substitution, scrambling)
- Dynamic path generation based on experiment parameters

**Real-time Monitoring:**
- Live GPU usage monitoring
- Experiment progress tracking
- Automatic result saving to structured directories

**Convenience Features:**
- Auto-generated Python commands for terminal execution
- Dynamic output directory creation
- Watermarked text path management
- Copy-paste ready command generation

### Basic Usage

1. **Select Experiment Type**: Choose between basic detection or robustness testing
2. **Configure Parameters**: Set algorithm, model, and dataset preferences
3. **Adjust Settings**: Modify watermark strength and sample size
4. **Execute**: Click "Execute Experiment" to run the configuration
5. **Monitor Results**: View real-time output and GPU usage
6. **Export**: Copy generated commands for batch processing

This web interface complements the command-line tools and provides an accessible entry point for researchers and practitioners working with LLM watermark detection.