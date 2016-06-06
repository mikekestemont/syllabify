#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Syllabify import Syllabifier
import utils

def main():
    """
    Example usage.
    """

    s = Syllabifier(nb_epochs = 30,
                    nb_layers = 3,
                    nb_dims = 50,
                    batch_size = 50,
                    model_dir = 'model_s')
    s.fit_transform_train_data(inp='../data/crm.txt',
                               max_nb = None)
    s.create_splits(test_prop=0.1, dev_prop=0.1)
    s.fit()

    # test save and reload:
    s.save()
    s = Syllabifier(model_dir = 'model_s',
                   load = True)

    # run Bouma et al:
    utils.run_bouma_et_al()

    # run the LSTM:
    s.syllabify(inp='../data/test_input.txt',
                outp='../data/lstm_output.txt')

    # evaluate both approaches:
    print('-> lstm scores:')
    s.evaluate(goldp='../data/test_gold.txt',
               silverp='../data/lstm_output.txt')

    print('-> Bouma et al scores:')
    s.evaluate(goldp='../data/test_gold.txt',
               silverp='../data/bouma_et_al_output.txt')

if __name__ == '__main__':
    main()