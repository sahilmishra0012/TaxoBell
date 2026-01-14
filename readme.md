# TaxoBell

Official Code of our paper Taxobell: Self Supervised Taxonomy expansion using Gaussian Box embeddings accepted in ACM WebConf 2026 Main Track.

## Datasets

We run experiments on the following datasets. You can add more datasets by following the exact format that we use. It is key to have the ```.taxo``` and ```.terms``` file separated to use our pipeline.

- Science
- Environment
- Wordnet
- MeSH
- Semeval Food 

## Running the code

```main.py``` is for training and ```main_pred.py``` is for inference or case studies.

You may download all data files from [here](https://drive.google.com/drive/folders/1qUv0PM14i1sqKgfVGKofnWCi3ymr6CMf?usp=sharing). These folders also contain prediction logs for your reference. To initiate the training process, execute the `main.py` script from the terminal. You may configure the training process by passing command-line arguments. For predictions and visualizations, the same hyperparameters are to be used along with the path of the model weights.

### 1. Basic Usage

Run the script with default hyperparameters. Ensure the `--dataset` and `--is_multi_parent` flags are set correctly for your target data.

```bash
python main.py --dataset environment --is_multi_parent False
```

### 2. Advanced Configuration

For a complete training run with specific hyperparameters, model selection, and Weights & Biases logging enabled, use the following format:

```bash
python main.py \
    --dataset computer_science \
    --is_multi_parent True \
    --model snowflake \
    --batch_size 64 \
    --epochs 100 \
    --lr 2e-5 \
    --negsamples 100 \
    --embed_size 64 \
    --method gated \
    --gpu_id 0 \
    --wandb 1
```

## Command Line Arguments

The following table details all available command-line arguments for configuration.

### Data and Model Configuration

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `environment` | The name of the target taxonomy dataset to be used for training. |
| `--model` | `str` | `bert` | The specific pre-trained language model backbone to initialize (e.g., 'bert', 'snowflake', 'e5'). |
| `--pre_train` | `str` | `bert` | Identifier string for the pre-training configuration, primarily used for experiment logging. |
| `--is_multi_parent` | `bool` | `False` | **Required.** Indicates if the dataset structure allows nodes to have multiple parents (DAG structure). |
| `--method` | `str` | `normal` | The projection method used to map embeddings to Gaussian parameters (e.g., 'normal', 'gated'). |
| `--complex` | `bool` | `False` | Enables complex-valued representations for quantum taxonomy modeling. |
| `--mixture` | `str` | `None` | Specifies the type of weighting strategy if using a mixture model approach. |

### Network Architecture

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--hidden` | `int` | `64` | The dimensionality of the hidden layers within the projection Multi-Layer Perceptrons (MLPs). |
| `--embed_size` | `int` | `8` | The dimensionality of the output Gaussian embeddings. |
| `--dropout` | `float` | `0.4` | The dropout probability applied during training for regularization. |
| `--padmaxlen` | `int` | `30` | The maximum token sequence length; sequences longer than this are truncated, and shorter ones are padded. |
| `--matrixsize` | `int` | `768` | The size of the density matrix when using specific complex/quantum configurations. |

### Training Hyperparameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` | The total number of training epochs to perform. |
| `--batch_size` | `int` | `128` | The number of samples per training batch. |
| `--lr` | `float` | `9e-5` | The learning rate for the optimizer. |
| `--lr_proj` | `float` | `1e-3` | The specific learning rate for the projection layers. |
| `--optim` | `str` | `adamw` | The optimization algorithm to use (e.g., 'adam', 'adamw'). |
| `--eps` | `float` | `1e-8` | The epsilon value for the AdamW optimizer to improve numerical stability. |
| `--accumulation_steps`| `int` | `1` | The number of steps to accumulate gradients before performing an optimizer step. |
| `--negsamples` | `int` | `50` | The number of negative samples generated per positive node pair. |
| `--seed` | `int` | `20` | The random seed used for initialization to ensure reproducibility. |

### Loss Function Weights

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--wtbce` | `float` | `0.45` | The weight assigned to the Bhattacharyya Coefficient loss component. |
| `--wtkl` | `float` | `0.45` | The weight assigned to the KL Divergence loss component. |
| `--wtreg` | `float` | `0.1` | The weight assigned to the regularization loss component. |
| `--lam` | `float` | `0.1` | The margin or weight scaling factor applied to the KL divergence containment term. |
| `--C` | `float` | `1.5` | A scaling constant used for the volume difference ratio in the loss function. |

### System and Logging

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--cuda` | `bool` | `True` | Enables CUDA for GPU acceleration. |
| `--gpu_id` | `int` | `1` | The index of the specific GPU to be used for training. |
| `--wandb` | `int` | `1` | Toggles Weights & Biases logging (1 for enabled, 0 for disabled). |
| `--expID` | `int` | `0` | An integer identifier for the experiment, used for file versioning and checkpointing. |