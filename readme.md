# neural-translation-transformer

RU-EN translation bases on Transformer model. It uses custom BPE tokenizer and 
gensim.Word2Vec model. 

## Models.

Transformer model bases on https://github.com/SamLynnEvans/Transformer

Attention head pruning bases on https://github.com/aiha-lab/Attention-Head-Pruning

# Data

All data you can find here: https://drive.google.com/drive/folders/1zVsotEzUDgA-j1SHhBPVeehO_AiA5Lq7?usp=sharing

This link contains pre-learn models, tokenizers, w2v models and dataset. You
can learn you own model, but dataset must be store in .tsv format, you also 
need to edit paths in **src/enums_and_constants/constants**.

# Training

## Tokenizer

First of all train you own tokenizer. Example:
```bash
python bpe-trainer.py
```
It will train and save source and target tokenizers bases on dataset.

## W2V model

After that you can train your own w2v model. You must train tokenizer before
w2v model cause it bases on tokens. Example:
```bash
python w2v-trainer.py
```
It will train and save source and target w2v models.

## Models

PyTorch Lightning used for model training. Each model trains 10 epoch with
early stop.

### Full model

Use
```bash
python model-trainer.py --prune False
```
to train model without attention head pruning. 

### Prune model

Use
```bash
python model-trainer.py --prune False
```
to train model with attention head pruning. By default, l0 coefficient equals 0.005

## Training results

