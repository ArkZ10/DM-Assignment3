"""Reproduce the top-scoring Kaggle submission for HAR Assignment 3.

Runs the final pseudo-labeling + augmentation pipeline (src/gen_pseudo_aug.py):
N_PASS1=5, N_PASS2=7, K_AUG_PSEUDO=1, THRESH=0.82, Viterbi decode.
Public LB score: 0.8061.

Output: /root/dm-assignment3/submission_pseudo_aug.csv

Usage:
    python main.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gen_pseudo_aug

if __name__ == "__main__":
    gen_pseudo_aug.main()
