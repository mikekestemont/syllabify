#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

import numpy as np

from sklearn.metrics import accuracy_score


def run_bouma_et_al(inp = '../data/test_input.txt',
                    outp = '../data/bouma_et_al_output.txt'):
    """
    * Runs via the command line the C-program by Bouma et al.
    * Will use the file `inp` as input, and `outp` as output.
    * The program needs to be compiled first, e.g.:
        >>> gcc -o bouma_et_al hyphenate_mnl.c
    """
    subprocess.call('./bouma_et_al < '+inp+' > '+outp, shell=True)

def load_data(fp = '../data/crm.txt', max_nb = None):
    """
    * Load words from the file `fp`.
    * These words can be syllabified or not.
    * For dev purposes, only `max_nb` (int) words 
      will be loaded, if this param is set.
    """
    tokens = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(line.lower())
            if max_nb:
                if len(tokens) >= max_nb:
                    break
    return tokens

def pred_to_classes(X):
    """
    * Convert the 3-dimensional representation of class labels
      (nb_words, nb_timesteps, 3) to a 2-dimensional representation
      of shape ((nb_words, nb_timesteps)).
    """
    words = []
    for w in X:
        words.append([np.argmax(p) for p in w])
    return np.array(words)

def metrics(G, S):
    """
    * Eats two results in a three-dimensional format (nb_words, nb_timesteps, 3)
      - `G`: ground truth
      - `D`: silver predictions by a system.
    * Returns evaluation metrics:
      - `hyphen_acc`: proportion of correctly classified
        characters
      - `token_acc`: proportion of  words which are completely
        accurately syllabified.
    * Dummy and padding characters outside words are ignored.
    """
    token_acc = 0.0
    g_hyphens, s_hyphens = [], []
    g_words, s_words = [], []
    for g, s in zip(G, S):
        idxs = g < 2
        g = tuple(g[idxs])
        s = tuple(s[idxs])
        if g == s:
            token_acc += (1. / len(G))
        g_hyphens.extend(list(g))
        s_hyphens.extend(list(s))

    hyphen_acc = accuracy_score(g_hyphens, s_hyphens)

    return token_acc * 100, hyphen_acc * 100

def stringify(orig_token, segmentation):
    """
    * Takes a original, unsyllabified `orig_token` (e.g. seruaes)
      and aligns it with a syllabification proposed (`segmentation`).
    * Returns the syllabified token in string format (e.g. ser-uaes).
    """
    orig_token = list(orig_token)
    s = segmentation[1 : len(orig_token) + 1]
    new_str = []
    for p in s[::-1]:
        if p == 0:
            new_str.append(orig_token[-1])
            del orig_token[-1]
        else:
            new_str.append(orig_token[-1])
            del orig_token[-1]
            new_str.append('-')
    return ''.join(new_str[::-1])