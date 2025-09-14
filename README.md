# Bizcompass Benchmark

A comprehensive benchmark for evaluating Large Language Models (LLMs) in business contexts including Economics, Finance, Operations Management, and Statistics.

## Project Structure

```
Bizcompass/
├── bizcompass.py              # Main entry point
├── config.py                  # Configuration management
├── config_example.yaml        # Example configuration file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── EVALUATION_CONFIG_GUIDE.md # Evaluation configuration guide
├── EVALUATION_PATH_GUIDE.md   # Evaluation path guide
│
├── Dataset/                   # Benchmark datasets
│   ├── Econ/                 # Economics questions
│   ├── Fin/                  # Finance questions
│   ├── OM/                   # Operations Management questions
│   └── Stat/                 # Statistics questions
│       ├── General QA/       # Open-ended questions
│       ├── Multiple Choice/  # Multiple choice questions
│       ├── Single Choice/    # Single choice questions
│       └── Table QA/         # Table-based questions
│
├── config/                   # Configuration files
├── dataloader/              # Data loading utilities
│   └── dataloader.py
├── evaluation/              # Evaluation modules
│   ├── evaluator.py         # LLM-based evaluator
│   └── metrics.py           # Metrics calculation
├── inference/               # Inference modules
│   ├── api_inference.py     # API-based inference
│   ├── debug_inference.py   # Debug inference
│   └── local_inference.py   # Local model inference
├── prompts/                 # Prompt management
│   ├── domain_prompts.py    # Domain-specific prompts
│   └── prompt_manager.py    # Prompt manager
├── scripts/                 # Utility scripts
│   └── batch_inference.py   # Batch processing
├── utils/                   # Utility functions
│   └── utils.py
├── results/                 # Output directory
│   ├── logs/               # Log files
│   └── result/             # Experiment results
└── logs/                   # Additional logs
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create a configuration file or use environment variables:

#### Option A: Environment Variables (Recommended)
```bash
export API_KEY="your-api-key-here"
export BASE_URL="https://api.openai.com/v1"  # or your API endpoint
```

#### Option B: Configuration File
Create `config.yaml`:
```yaml
api:
  api_key: "your-api-key-here"
  base_url: "https://api.openai.com/v1"

model:
  model_name: "gpt-4o"
  temperature: 0.0
  top_p: 0.95

processing:
  domains: ["Econ", "Fin", "OM", "Stat"]
  question_types: ["Single Choice", "Multiple Choice", "General QA", "Table QA"]
```

### 3. Run Inference

#### Basic Inference
```bash
python bizcompass.py inference --model_name gpt-4o
```

#### Custom Configuration
```bash
python bizcompass.py inference \
  --model_name claude-3-sonnet \
  --domains "Econ,Fin" \
  --question_types "Single Choice,Multiple Choice" \
  --temperature 0.8 \
  --top_p 0.95
```

#### Debug Mode (No API calls, mock responses)
```bash
python bizcompass.py inference --model_name gpt-4o --debug
```

### 4. Run Full Pipeline (Inference + Evaluation)

The pipeline command runs both inference and evaluation in sequence, providing immediate score results:

#### Basic Pipeline
```bash
python bizcompass.py pipeline --model_name gpt-4o
```

#### Pipeline with Debug Mode
```bash
python bizcompass.py pipeline \
  --model_name test-model \
  --debug \
  --domains "Econ" \
  --question_types "Single Choice"
```

#### Custom Pipeline Configuration
```bash
python bizcompass.py pipeline \
  --model_name claude-3-sonnet \
  --domains "Econ,Fin" \
  --question_types "Single Choice,General QA" \
  --temperature 0.8 \
  --evaluator_model "gpt-4o"
```

The pipeline will:
1. 🚀 Run inference on the specified model
2. 📊 Run evaluation using the specified evaluator model
3. 🎯 Display final scores:
   - **Choice Questions**: Accuracy percentage
   - **Open-ended Questions**: Average score (0.0-1.0)

## Configuration Guide

### API Configuration

For API-based models (OpenAI, Anthropic, etc.):

```yaml
api:
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"  # or your endpoint
  max_workers: 20
  request_timeout: 60
```

### Model Configuration

```yaml
model:
  model_type: "api"  # or "local"
  model_name: "gpt-4o"
  temperature: 0.0    # 0.0 for deterministic, 0.8 for creative
  top_p: 0.95
  max_tokens_choice: 512
  max_tokens_general: 8192
```

### Processing Configuration

```yaml
processing:
  domains: ["Econ", "Fin", "OM", "Stat"]
  question_types: ["Single Choice", "Multiple Choice", "General QA", "Table QA"]
  max_workers: 20
```

### Local Model Configuration

For local models (Llama, etc.):

```yaml
model:
  model_type: "local"
  model_name: "llama3-8b-instruct"

paths:
  model_path: "/path/to/your/model"

batch:
  default_batch_size: 64
  enable_oom_handling: true
```

## Usage Examples

### 1. Single Model Inference

```bash
# Basic inference with default settings
python bizcompass.py inference --model_name gpt-4o

# Custom domains and question types
python bizcompass.py inference \
  --model_name claude-3-sonnet \
  --domains "Econ" \
  --question_types "Single Choice" \
  --temperature 0.0
```

### 2. Batch Processing

```bash
# Run multiple models with different parameters
python scripts/batch_inference.py \
  --models "gpt-4o,claude-3-sonnet" \
  --temperatures "0.0,0.8" \
  --top_ps "0.95"
```

### 3. Batch Evaluation

```bash
# Evaluate existing inference results
python scripts/batch_evaluation_script.py \
  --models "gpt-4o,claude-3-sonnet" \
  --domains "Econ,Fin" \
  --question_types "Single Choice,Multiple Choice" \
  --evaluator_model "gpt-4o"

# List available inference results
python scripts/batch_evaluation_script.py --list_results

# Dry run to see what would be evaluated
python scripts/batch_evaluation_script.py --dry_run --models "gpt-4o"
```

### 3. Full Pipeline (Inference + Evaluation)

```bash
# Run inference and evaluation together
python bizcompass.py pipeline \
  --model_name gpt-4o \
  --evaluator_model claude-3-sonnet
```

### 4. Debug Mode

```bash
# Test without API calls
python bizcompass.py inference --model_name gpt-4o --debug

# Debug specific components
python bizcompass.py inference --model_name gpt-4o --debug_inference
python bizcompass.py evaluation --model_name gpt-4o --debug_evaluation
```

## Command Line Options

### Inference Command
```bash
python bizcompass.py inference [OPTIONS]

Options:
  --model_name TEXT          Model name (required)
  --model_type [api|local]   Model type (default: api)
  --domains TEXT             Comma-separated domains
  --question_types TEXT      Comma-separated question types
  --temperature FLOAT        Sampling temperature (default: 0.0)
  --top_p FLOAT             Top-p sampling (default: 0.95)
  --debug                    Enable debug mode
  --api_key TEXT            API key
  --base_url TEXT           API base URL
```

### Evaluation Command
```bash
python bizcompass.py evaluation [OPTIONS]

Options:
  --model_name TEXT          Model to evaluate (required)
  --evaluator_model TEXT     Evaluator model (default: gpt-4o)
  --domains TEXT             Comma-separated domains
  --question_types TEXT      Comma-separated question types
  --debug                    Enable debug mode
```

### Pipeline Command
```bash
python bizcompass.py pipeline [OPTIONS]

Options:
  --model_name TEXT          Model name (required)
  --model_type [api|local]   Model type (default: api)
  --domains TEXT             Comma-separated domains
  --question_types TEXT      Comma-separated question types
  --temperature FLOAT        Sampling temperature
  --top_p FLOAT             Top-p sampling
  --skip_evaluation          Skip evaluation step
  --debug                    Enable debug mode
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for API models | - |
| `BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `BIZCOMPASS_TEMPERATURE` | Default temperature | `0.0` |
| `BIZCOMPASS_TOP_P` | Default top-p | `0.95` |
| `BIZCOMPASS_DOMAINS` | Default domains | `Econ,Fin,OM,Stat` |
| `BIZCOMPASS_QUESTION_TYPES` | Default question types | `Single Choice,Multiple Choice,General QA,Table QA` |

## Output Structure

Results are saved in the `results/` directory:

```
results/
├── inference/
│   └── inference_model_name_timestamp/
│       ├── config.yaml      # Complete configuration
│       ├── prompts.json     # All prompts used
│       ├── results.json     # Experiment results
│       └── model_output/    # Original model outputs
└── logs/
    └── inference/
```

## Supported Models

### API Models
- OpenAI: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-sonnet`, `claude-3-opus`, `claude-3-haiku`
- Google: `gemini-pro`, `gemini-pro-vision`
- Others: Any OpenAI-compatible API

### Local Models
- Llama 3: `llama3-8b-instruct`, `llama3-70b-instruct`
- Other Hugging Face models with chat templates

## Dataset

The benchmark includes questions from four business domains:

- **Economics (Econ)**: Microeconomics, macroeconomics, economic theory
- **Finance (Fin)**: Corporate finance, investments, financial markets
- **Operations Management (OM)**: Supply chain, logistics, operations
- **Statistics (Stat)**: Statistical analysis, probability, data science

Each domain includes four question types:
- **Single Choice**: Multiple choice with one correct answer
- **Multiple Choice**: Multiple choice with multiple correct answers
- **General QA**: Open-ended questions requiring detailed answers
- **Table QA**: Questions based on tabular data

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   export API_KEY="your-actual-api-key"
   ```

2. **Model Not Found**
   - Check model name spelling
   - Verify API endpoint supports the model

3. **Out of Memory (Local Models)**
   - Reduce batch size: `--batch_size 32`
   - Use smaller model or enable OOM handling

4. **Rate Limiting**
   - Reduce `max_workers` in configuration
   - Add delays between requests

### Debug Mode

Use debug mode to test without API calls:

```bash
python bizcompass.py inference --model_name gpt-4o --debug
```

This will:
- Show colored output with progress
- Use mock responses instead of API calls
- Display processing statistics
- Skip experiment recording

## Examples

### Example 1: Basic API Inference
```bash
# Set API key
export API_KEY="sk-your-openai-key"

# Run inference
python bizcompass.py inference --model_name gpt-4o
```

### Example 2: Custom Configuration
```bash
# Create config.yaml
cat > config.yaml << EOF
api:
  api_key: "your-key"
  base_url: "https://api.openai.com/v1"

model:
  model_name: "gpt-4o"
  temperature: 0.8

processing:
  domains: ["Econ", "Fin"]
  question_types: ["Single Choice"]
EOF

# Run with config
python bizcompass.py inference --model_name gpt-4o
```

### Example 3: Local Model
```bash
# Set model path
export MODEL_PATH="/path/to/llama3-8b-instruct"

# Run local inference
python bizcompass.py inference \
  --model_name llama3-8b-instruct \
  --model_type local
```

### Example 4: Debug Testing
```bash
# Test without API calls
python bizcompass.py inference \
  --model_name gpt-4o \
  --domains "Econ" \
  --question_types "Single Choice" \
  --debug
```

## Advanced Usage

### Batch Processing
```bash
python scripts/batch_inference.py \
  --models "gpt-4o,claude-3-sonnet,gemini-pro" \
  --temperatures "0.0,0.8" \
  --domains "Econ,Fin" \
  --question_types "Single Choice,Multiple Choice"
```

### Custom Prompts
Modify prompts in `prompts/domain_prompts.py` for custom behavior.

## Contributing

1. Fork the repository
2. Create a feature branch  
3. Make your changes
4. Test with debug mode
5. Submit a pull request

## License

This project is licensed under the MIT License.