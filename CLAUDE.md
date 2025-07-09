# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements HMamba, a Computer-Assisted Pronunciation Training (CAPT) system using Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss for pronunciation assessment. The system evaluates pronunciation at phone, word, and utterance levels.

## Essential Commands

### Environment Setup
```bash
# Setup environment (must be done first)
nano/vim path.sh  # Edit KALDI_ROOT and conda path
source path.sh    # Load environment
```

### Training and Testing
```bash
# Main training pipeline
bash run.sh

# Individual stages
bash run.sh --stage 1  # Training only
bash run.sh --stage 2  # Generate phone transcripts
bash run.sh --stage 3  # Evaluate MDD metrics

# Manual training with custom parameters
CUDA_VISIBLE_DEVICES=0 python traintest.py \
  --phn-dict local/so762/vocab_merge.json \
  --seed 824 \
  --lr 2e-3 \
  --batch-size 50 \
  --n-epochs 20 \
  --model HMamba \
  --model-conf conf/so762/HMamba.yaml \
  --exp-dir exp/so762/hmamba/0
```

### Recognition and Evaluation
```bash
# Generate recognition results
python recog.py --model HMamba --model-conf conf/so762/HMamba.yaml --exp-dir exp/so762/hmamba/0

# Evaluate MDD metrics
eval_mdd/mdd_result.sh exp/so762/hmamba/0/rel_nosil exp/so762/hmamba/0/can_nosil exp/so762/hmamba/0/hyp_nosil

# Collect results across multiple runs
python collect_summary.py --exp-dir exp/so762/hmamba
python collect_mdd.py --exp-dir exp/so762/hmamba
```

### Development Tools
```bash
# Install dependencies
pip install -r requirements.txt

# Check experiment results
cat exp/so762/hmamba/result_mdd.txt
```

## Architecture Overview

### Core Components

#### HMamba Model (`models/hmamba.py`)
- **Hierarchical Architecture**: 5 processing blocks (3 phone-level, 1 word-level, 1 utterance-level)
- **BiMamba Blocks**: Uses bidirectional Mamba (selective state space model) for sequence modeling
- **Multi-level Prediction**: Outputs phone scores, word scores (accuracy/stress/total), and utterance scores (accuracy/completeness/fluency/prosodic/total)
- **Feature Fusion**: Combines GOP features, SSL features (wav2vec2, HuBERT, WavLM), and raw audio features

#### Dataset (`dataset.py`)
- **GoPDataset**: Loads GOP features, SSL features, and raw audio features
- **Multi-modal Input**: Handles GOP (84-dim), SSL (3x1024-dim), and raw audio (8-dim) features
- **Normalization**: Applies AM-specific normalization (LibriSpeech: mean=3.203, std=4.045)

#### Loss Functions (`loss.py`)
- **Decoupled Cross-entropy**: Separates correct and mispronounced phonemes with weighted loss
- **Multi-task Loss**: Combines phone MSE, word MSE, utterance MSE, and classification losses

### Training Pipeline (`traintest.py`)
- **Multi-level Training**: Simultaneous training for phone, word, and utterance level assessments
- **Tri-stage Scheduler**: Custom learning rate scheduling with warmup, peak, and decay phases
- **Wandb Integration**: Automatic experiment tracking
- **Early Stopping**: Based on phone-level MSE performance

### Data Structure
```
data/so762/
├── gop-librispeech-bies/     # GOP features
├── wav2vec2-large-xlsr-53/   # SSL features
├── hubert-large-ll60k/       # SSL features  
├── wavlm-large/              # SSL features
└── raw-audio/                # Raw audio features
```

### Model Configuration (`conf/so762/HMamba.yaml`)
- **Embedding**: 128-dim with positional and canonical phone embeddings
- **Mamba Parameters**: d_state=16, d_conv=2, expand=4
- **Pooling**: Score-attention pooling for utterance-level aggregation
- **Vocabulary**: 49 phonemes (excluding PAD token)

## Key Features

### Multi-modal Input Processing
- GOP features: Goodness of Pronunciation features
- SSL features: Self-supervised learning features from multiple models
- Raw audio: Duration and energy features

### Hierarchical Assessment
- **Phone Level**: Individual phoneme pronunciation quality
- **Word Level**: Word-level accuracy, stress, and overall scores
- **Utterance Level**: Overall pronunciation assessment across multiple dimensions

### Decoupled Loss Design
- Separates correct and mispronounced phonemes in loss calculation
- Weighted combination with configurable parameters (loss_w_a, loss_w_xent)

## Important Notes

- **Environment Dependencies**: Requires Kaldi installation and specific CUDA environment
- **Data Preparation**: Features must be pre-computed and placed in expected directory structure
- **Model Variants**: Currently supports HMamba model with bimamba or transformer blocks
- **Evaluation**: Uses Phone Error Rate (PER) and correlation metrics for assessment