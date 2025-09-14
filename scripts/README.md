# Scripts Directory

This directory contains utility scripts for batch processing and automation in the Bizcompass benchmark.

## Available Scripts

### batch_evaluation_script.py

Batch evaluation script for processing multiple evaluation files in specified directories.

#### Usage

```bash
# Run from the Bizcompass root directory
python scripts/batch_evaluation_script.py --base_path "results/result/Econ/general/llama3-8b-instruct"

# Or run from the scripts directory
cd scripts
python batch_evaluation_script.py --base_path "../results/result/Econ/general/llama3-8b-instruct"
```

#### Command Line Arguments

- `--base_path`: Base path for evaluation (required)
- `--evaluator_model`: Evaluator model (default: "gpt-4o")
- `--max_workers`: Maximum number of workers (default: 5)
- `--domains`: Comma-separated list of domains
- `--question_types`: Comma-separated list of question types
- `--pattern`: File matching pattern (default: "questions_eval.json")
- `--debug`: Enable debug mode

#### Examples

```bash
# Evaluate all files in a specific model directory
python scripts/batch_evaluation_script.py --base_path "results/result/Econ/general/llama3-8b-instruct"

# Evaluate by domain and question type
python scripts/batch_evaluation_script.py --base_path "results/result" --domains "Econ,Fin" --question_types "General QA"

# Use debug mode with custom evaluator model
python scripts/batch_evaluation_script.py --base_path "results/result" --debug --evaluator_model "test_model"

# Increase concurrency
python scripts/batch_evaluation_script.py --base_path "results/result" --max_workers 10
```

#### Programmatic Usage

```python
from scripts.batch_evaluation_script import BatchEvaluator
from config import create_default_config

# Create configuration
config = create_default_config(debug_evaluation=True)

# Create batch evaluator
batch_evaluator = BatchEvaluator(config)

# Batch evaluate directory
results = batch_evaluator.batch_evaluate_directory(
    base_path=Path("results/result/Econ/general/llama3-8b-instruct"),
    evaluator_model="gpt-4o",
    max_workers=5
)
```

## Notes

- All scripts are designed to be run from the Bizcompass root directory
- The scripts automatically handle relative path resolution
- Debug mode bypasses LLM calls and uses mock responses
- Results are automatically saved to JSON files
