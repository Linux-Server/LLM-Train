# Supervised Fine-Tuning (SFT)

This directory contains resources and notebooks for Supervised Fine-Tuning (SFT) of Large Language Models using Hugging Face.

## Overview

Supervised Fine-Tuning is a technique used to adapt pre-trained language models to specific tasks or domains by training them on labeled datasets. This approach helps improve model performance for targeted use cases while leveraging the knowledge already learned during pre-training.

## Contents

- `1.ipynb` - Main notebook for SFT training and experimentation

## What is Supervised Fine-Tuning?

SFT involves:
- Taking a pre-trained language model
- Training it on a supervised dataset (input-output pairs)
- Optimizing the model to better perform on specific tasks

## Common Use Cases

- **Instruction Following**: Training models to follow specific instructions
- **Task Specialization**: Adapting models for specific domains (medical, legal, coding, etc.)
- **Behavior Alignment**: Aligning model outputs with desired behaviors
- **Custom Applications**: Creating models tailored to specific business needs

## Prerequisites

Before running the notebooks, ensure you have:

```bash
pip install transformers datasets accelerate peft bitsandbytes torch
```

### Required Libraries

- `transformers` - Hugging Face Transformers library
- `datasets` - For loading and processing datasets
- `accelerate` - For distributed training and optimization
- `peft` - Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
- `bitsandbytes` - For quantization and memory optimization
- `torch` - PyTorch framework

## Getting Started

1. **Prepare Your Dataset**: Ensure your dataset is in the correct format (typically instruction-response pairs)
2. **Configure Training Parameters**: Set hyperparameters like learning rate, batch size, and number of epochs
3. **Run Training**: Execute the notebook to fine-tune your model
4. **Evaluate Results**: Test the fine-tuned model on validation data
5. **Save and Deploy**: Export the model for inference

## Key Concepts

### LoRA (Low-Rank Adaptation)
A parameter-efficient fine-tuning technique that adds trainable rank decomposition matrices to existing weights, significantly reducing memory requirements.

### QLoRA (Quantized LoRA)
Combines quantization with LoRA for even more efficient fine-tuning, enabling training of large models on consumer hardware.

### Training Hyperparameters
- **Learning Rate**: Controls the step size during optimization
- **Batch Size**: Number of samples processed before updating weights
- **Epochs**: Number of complete passes through the training dataset
- **Max Length**: Maximum sequence length for tokenization

## Best Practices

1. **Start Small**: Begin with a smaller model or subset of data to validate your approach
2. **Monitor Metrics**: Track loss, perplexity, and task-specific metrics during training
3. **Use Validation Set**: Always evaluate on held-out data to prevent overfitting
4. **Save Checkpoints**: Regularly save model checkpoints during training
5. **Version Control**: Track dataset versions and training configurations

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Trainer Guide](https://huggingface.co/docs/transformers/main_classes/trainer)

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Try quantization (8-bit or 4-bit)

### Slow Training
- Enable mixed precision training (fp16 or bf16)
- Use Flash Attention 2
- Optimize DataLoader settings
- Consider multi-GPU training

### Poor Results
- Check data quality and formatting
- Adjust learning rate
- Increase training epochs
- Verify model selection for your task

## Contributing

When adding new notebooks or examples:
1. Document all hyperparameters
2. Include example outputs
3. Add error handling
4. Provide clear comments and markdown cells

## License

Please refer to the main repository LICENSE file.
