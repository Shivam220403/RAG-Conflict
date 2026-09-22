# Purpose : Download all 4 natural datasets from HuggingFace and inspect them
# Author  : Shivam

# IMPORTS
from datasets import load_dataset
import os
import json


# SETUP: CREATE FOLDERS TO STORE RAW DATA
os.makedirs("raw/wiki_contradict", exist_ok=True)
os.makedirs("raw/nq_open",         exist_ok=True)
os.makedirs("raw/llmbar",          exist_ok=True)
os.makedirs("raw/ragtruth",        exist_ok=True)

# DATASET 1: WikiContradict
# Role    : The CONFLICT SET — real contradictions from Wikipedia
# Size    : 253 human-annotated pairs

print("\n" + "="*60)
print("LOADING: WikiContradict")
print("="*60)

wiki = load_dataset("ibm-research/Wikipedia_contradict_benchmark")

# INSPECT
print("\n[Structure]")
print(wiki)                          # shows all splits and their sizes

print("\n[Column Names]")
print(wiki['train'].column_names)    # shows what fields each row has

print("\n[First Example — Row 0]")
print(wiki['train'][0])              # prints the very first data point

print("\n[Total rows]", len(wiki['train']))


# DATASET 2: NQ-Open (Natural Questions)
# Role    : The CONTROL SET — standard questions with one clean context
# Size    : 3000+ (we will use 500)

print("\n" + "="*60)
print("LOADING: NQ-Open")
print("="*60)

nq = load_dataset("google-research-datasets/nq_open")

print("\n[Structure]")
print(nq)

print("\n[Column Names]")
print(nq['train'].column_names)

print("\n[First Example — Row 0]")
print(nq['train'][0])

print("\n[Total rows in train]", len(nq['train']))


# DATASET 3: LLMBar
# Role    : The JUDGE AUDITOR — tests bias in LLM evaluators
# Size    : Adversarial subset

print("\n" + "="*60)
print("LOADING: LLMBar")
print("="*60)

# This dataset has multiple subsets — 'Adversarial' is the one we need
llmbar = load_dataset("princeton-nlp/LLMBar", "Adversarial")

print("\n[Structure]")
print(llmbar)

print("\n[Column Names]")
# Some datasets only have 'test' split, not 'train'
first_split = list(llmbar.keys())[0]   # get whatever split exists
print(llmbar[first_split].column_names)

print("\n[First Example — Row 0]")
print(llmbar[first_split][0])

print("\n[Total rows]", len(llmbar[first_split]))


# DATASET 4: RAGTruth
# Role    : The CALIBRATION SET — word-level RAG hallucination examples
# Size    : ~18,000 examples
# Note    : This one is on HuggingFace (mirrored from GitHub)

print("\n" + "="*60)
print("LOADING: RAGTruth")
print("="*60)

ragtruth = load_dataset("wandb/RAGTruth")   # community mirror

print("\n[Structure]")
print(ragtruth)

print("\n[Column Names]")
first_split = list(ragtruth.keys())[0]
print(ragtruth[first_split].column_names)

print("\n[First Example — Row 0]")
print(ragtruth[first_split][0])

print("\n[Total rows]", len(ragtruth[first_split]))


# SUMMARY
print("\n" + "="*60)
print("ALL DATASETS LOADED SUCCESSFULLY")
print("Next Step: Inspect column names above, then write standardize.py")
print("="*60)

