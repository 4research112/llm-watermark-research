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
