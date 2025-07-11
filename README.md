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
# Activate the markllm environment
conda activate markllm
```

2. **Install dependencies**

```bash
# If the environment doesn't exist yet, create it first and install dependencies
# conda create -n markllm python=3.10
# conda activate markllm
# pip install -r requirements.txt
```

## Evaluation Pipelines

### Single Experiment Execution

Run a single experiment configuration:

```bash
python3 script/paraphraser.py \
    --algorithm KGW \
    --max_samples 1000 \
    --output_dir tables_data_100/llama3.1/kgw/enc4_d2 \
    --dataset dataset/c4/processed_c4.json \
    --generation_mode generate \
    --delta 2.0
```

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

**Key Parameters:**

- `--algorithm`: Watermarking algorithm (KGW, SWEET, Unigram, EXP)
- `--max_samples`: Number of experiment samples
- `--delta`: Watermark strength parameter
- `--dataset`: Dataset path
- `--generation_mode`: Mode selection (generate/load)

<!-- ## Results -->

<!-- Experimental results tables -->

## Models and Datasets

**Models Used for Experiments:**
- facebook/opt-1.3b
- meta-llama/Llama-3.1-8B-Instruct

**Supported Datasets:**
- C4 Dataset
- WMT16 DE-EN
- HumanEval
- MBPP (Mostly Basic Python Problems)
- Traditional Chinese Dataset
- ZHTW C4 Dataset

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
- Choose models (Llama-3.1, OPT-1.3b, TAIDE)
- Configure datasets (C4, ZHTW, HumanEval, MBPP)
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

### Output Management

The interface automatically organizes results using structured paths:
- Output directories: `tables_data_{samples}/{model}/{algorithm}/{dataset}_{params}/`
- Watermark texts: `texts1000/{model}/{algorithm}/{dataset}_{params}/watermarked_texts.json`
- Results saved to: `{output_dir}/res.txt`

This web interface complements the command-line tools and provides an accessible entry point for researchers and practitioners working with LLM watermark detection.