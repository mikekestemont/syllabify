#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import shutil

import numpy as np

from sklearn.cross_validation import train_test_split

from keras.models import model_from_json
from keras.models import Model
from keras.utils import np_utils
from keras.layers import *

import utils


class Syllabifier:

    def __init__(self, max_input_len = None, nb_epochs = 30,
                 nb_layers = 3, nb_dims = 50, load = False,
                 batch_size = 50, model_dir = 'model_s'):
        self.max_input_len = max_input_len
        self.nb_epochs = nb_epochs
        self.nb_layers = nb_layers
        self.nb_dims = nb_dims
        self.batch_size = batch_size
        self.model_dir = model_dir

        if self.model_dir and not load:
            # make sure we have a clean model dir:
            if os.path.isdir(self.model_dir):
                shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

        elif self.model_dir and load:
            self.model = model_from_json(\
                open(os.sep.join([self.model_dir, 'model_architecture.json'])).read())
            self.model.load_weights(os.sep.join([self.model_dir, 'weights.h5']))
            self.model.compile(optimizer='RMSprop',
                          loss={'out': 'categorical_crossentropy'},
                          metrics=['accuracy'])
            self.char_lookup = pickle.load(\
                open(os.sep.join([self.model_dir, 'char_lookup.p']), 'rb'))
            self.max_input_len = pickle.load(\
                open(os.sep.join([self.model_dir, 'max_input_len.p']), 'rb'))
            self.filler = np.zeros(len(self.char_lookup.keys()))

    def save(self):
        """
        * Pickle all necessary objects to `model_dir`
          for later re-use.
        * Note: this won't save the weights,
          which are stored by the `fit() function!
        """
        open(os.sep.join([self.model_dir, 'model_architecture.json']), 'w')\
            .write(self.model.to_json())
        self.model.save_weights(\
            os.sep.join([self.model_dir, 'weights.h5']), overwrite=True)
        pickle.dump(self.char_lookup, \
            open(os.sep.join([self.model_dir, 'char_lookup.p']), 'wb'))
        pickle.dump(self.max_input_len, \
            open(os.sep.join([self.model_dir, 'max_input_len.p']), 'wb'))
        pickle.dump(self.char_lookup, \
            open(os.sep.join([self.model_dir, 'char_lookup.p']), 'wb'))


    def fit_transform_train_data(self, data = None, inp = None,
                                 max_input_len=None, max_nb = None):
        """
        * Vectorizes the training data passed, needs either:
            - `data`: a list of strings (['serv-vaes', 'ick', ..., 'ue-le'])
            - `inp`: path to a file with one item per line
            - for dev purposes, a `max_nb` (int) can be specified,
              so that only the first `max_nb` of words are loaded.
        * `max_input_len` determines the maximum length of
            input strings, i.e. the spatial dimensionality
            of the LSTM layers. If this is not explicitly set,
            it will use the length of the longest input string.
        * Vectorization conventions:
            - each word gets a special character prepended at
              the beginning (`%`) and end (`@`).
            - words get padded to length `max_input_len` re-using
              an all-zero filler (represented as `PAD`).
        * Class labels:
            - `0`: plain characters inside word
            - `1`: syllable-initial character
            - `2`: dummy filler
        """

        if inp:
            data = utils.load_data(inp, max_nb = max_nb)

        tokens, segmentations = [], []
        for token in data:
            chars, labels = [], []
            for idx, char in enumerate(token):
                if char != '-':
                    chars.append(char)
                else:
                    continue
                if idx == 0:
                    labels.append(0)
                else:
                    if token[idx - 1] == '-':
                        labels.append(1) # beginning of syllable
                    else:
                        labels.append(0)
            tokens.append(chars)
            segmentations.append(labels)
        
        # determine max len:
        if not self.max_input_len:
            self.max_input_len = len(max(tokens, key=len))
        print('Longest input token:', self.max_input_len)

        # we determine the vocabulary:
        self.char_vocab = tuple(sorted(set([item for sublist in tokens for item in sublist])))

        # we add symbols for begin, end and padding:
        self.char_vocab += tuple(('%', '@'))
        print('Character vocabulary:', ' '.join(self.char_vocab))

        # assign one-hot vector to each char for
        # fast vectorization:
        self.filler = np.zeros(len(self.char_vocab)) 
        self.char_lookup = {} 
        for idx, char in enumerate(self.char_vocab):
            char_vector = self.filler.copy()
            char_vector[idx] = 1.0
            self.char_lookup[char] = char_vector

        # uniformize len of of the outputs:
        outputs = []
        for segm in segmentations:
            segm = [2] + segm + [2]
            while len(segm) < (self.max_input_len + 2):
                segm.append(2)
            outputs.append(np_utils.to_categorical(segm, nb_classes=3))
        self.Y_train = np.array(outputs, dtype='int8')
        print('output shape:', self.Y_train.shape)

        X = []
        for token in tokens:
            token = ['%'] + token + ['@'] # add beginning
            while len(token) < (self.max_input_len + 2):
                token.append('PAD')
            x = []
            for char in token:
                try:
                    x.append(self.char_lookup[char])
                except KeyError:
                    x.append(self.filler)
            X.append(x)

        self.X_train = np.array(X, dtype='float32')
        print('Shape of the vectorized input:', self.X_train.shape)

        self.train_tokens = tokens

    def create_splits(self, test_prop=.1, dev_prop=.1):
        """
        Takes the train data and creates train-dev-test splits:
            * if a `test_prop` (float) is passed, a test set
              will be created and saved (for later comparisons).
            * if a `dev_prop` (float) is passed, a dev set will
              be created on the basis of the training material 
              which remains after creating the test set.
        """
        if test_prop:
            self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.train_tokens, self.test_tokens = \
                train_test_split(self.X_train, self.Y_train,\
                    self.train_tokens, test_size=dev_prop, random_state=42982)

            # save test data for comparison to Bouma et al:
            gt = utils.pred_to_classes(self.Y_test)
            with open('../data/test_gold.txt', 'w') as f:
                for i in [utils.stringify(o, p) for o, p in \
                            zip(self.test_tokens, gt)]:
                    f.write(i+'\n')
            with open('../data/test_input.txt', 'w') as f:
                for i in self.test_tokens:
                    f.write(''.join(i)+'\n')

        if dev_prop:
            self.X_train, self.X_dev, self.Y_train, self.Y_dev,\
                self.train_tokens, self.dev_tokens = \
                train_test_split(self.X_train, self.Y_train,\
                    self.train_tokens, test_size=dev_prop, random_state=4767)


    def build_model(self):
        """
        * Defines a stacked, bidirectrional LSTM model,
          which learns to map `max_input_len` + 2 characters
          to `max_input_len` + 2 labels.
        * Model parameters should be set in the general constructor, mainly:
            - `nb_epochs` (int) = maximum # epochs to train the model
            - `nb_layers` (int) = # number of stacked LSTM
            - `nb_dims` (int) = dimensionality of the LSTMs
            - `batch_size` = size of the minibatches used

        The model uses the RMSprop mechanism for gradient updates.
        """
        char_input = Input(shape=(self.max_input_len + 2, len(self.char_vocab)),
                            name='char_input')

        for i in range(self.nb_layers):

            if i == 0:
                curr_input = char_input
            else:
                curr_input = curr_enc_out

            # following block only runs on TF, bug in Theano?
            curr_enc_out = Bidirectional(LSTM(output_dim=self.nb_dims,
                                              return_sequences=True,
                                              activation='tanh',
                                              name='enc_lstm_'+str(i + 1)),
                                         merge_mode='sum')(curr_input)
            """
            
            #old, buggy version > now using bidirectional wrapper
            l2r = LSTM(output_dim=self.nb_dims,
                       return_sequences=True,
                       activation='tanh',
                       name='left_enc_lstm_'+str(i + 1))(curr_input)
            r2l = LSTM(output_dim=self.nb_dims,
                       return_sequences=True,
                       activation='tanh',
                       go_backwards=True,
                       name='right_enc_lstm_'+str(i + 1))(curr_input)
            curr_enc_out = merge([l2r, r2l], name='encoder_'+str(i+1), mode='sum')
            """


        dense = TimeDistributed(Dense(3), name='dense')(curr_enc_out)
        segm_out = Activation('softmax', name='out')(dense)

        self.model = Model(input=char_input, output=segm_out)
        print('Compiling model...')
        self.model.compile(optimizer='RMSprop',
                      loss={'out': 'categorical_crossentropy'},
                      metrics=['accuracy'])
        print('Model compiled!')

    def fit(self):
        """
        * Fits the model during x `nb_epochs`.
        * If dev data is available, dev scores will be
          calculated after each epoch.
        * If test data is available, test scores are
          calculated after the fitting process.
        * Only the model weights which reach the highest
          hyphenation accuracy are eventually stored.
        """
        if not hasattr(self, 'model'):
            self.build_model()

        train_inputs = {'char_input': self.X_train}
        train_outputs = {'out': self.Y_train}

        if hasattr(self, 'X_dev'):
            dev_inputs = {'char_input': self.X_dev}

        best_acc = [0.0, 0]

        for e in range(self.nb_epochs):
            print('-> epoch', e + 1)
            self.model.fit(train_inputs, train_outputs,
                      nb_epoch = 1,
                      shuffle = True,
                      batch_size = self.batch_size,
                      verbose=1)

            preds = self.model.predict(train_inputs,
                      batch_size = self.batch_size,
                      verbose=0)

            token_acc, hyphen_acc = utils.metrics(utils.pred_to_classes(self.Y_train),
                                            utils.pred_to_classes(preds))
            print('\t- train scores:')
            print('\t\t + token acc:', round(token_acc, 2))
            print('\t\t + hyphen acc:', round(hyphen_acc, 2))

            if hasattr(self, 'X_dev'):
                preds = self.model.predict(dev_inputs,
                           batch_size = self.batch_size,
                           verbose=0)

                token_acc, hyphen_acc = utils.metrics(utils.pred_to_classes(self.Y_dev),
                                                utils.pred_to_classes(preds))

                print('\t- dev scores:')
                print('\t\t + token acc:', round(token_acc, 2))
                print('\t\t + hyphen acc:', round(hyphen_acc, 2))

                if hyphen_acc > best_acc[0]:
                    print('\t-> saving weights')
                    self.model.save_weights(\
                        os.sep.join([self.model_dir, 'weights.h5']), overwrite=True)
                    best_acc = [hyphen_acc, e]

        # make sure we have the best weights:
        print('-> Optimal dev hyphenation accuracy:', round(best_acc[0],2),
              'at epoch #', best_acc[1])
        self.model.load_weights(os.sep.join([self.model_dir, 'weights.h5']))

        if hasattr(self, 'X_test'):
            test_inputs = {'char_input': self.X_test}

            preds = self.model.predict(test_inputs,
                           batch_size = self.batch_size,
                           verbose=0)
            token_acc, hyphen_acc = utils.metrics(utils.pred_to_classes(self.Y_test),
                                            utils.pred_to_classes(preds))
            print('\t- test scores:')
            print('\t\t + token acc:', round(token_acc, 2))
            print('\t\t + hyphen acc:', round(hyphen_acc, 2))

    def syllabify(self, data = None, inp = None, outp = None):
        """
        * Eats new, unsyllabified words either as a list (`data`)
          or from a file (`inp`)
        * Returns the syllabified words in string format
          (e.g. ser-uaes) and saves these to a file if `outp`
          is specified.
        """
        if inp:
            data = utils.load_data(inp)

        X = []
        for token in data:
            token = list(token)[:self.max_input_len]
            token = ['%'] + token + ['@'] # add beginning
            while len(token) < (self.max_input_len + 2):
                token.append('PAD')
            x = []
            for char in token:
                try:
                    x.append(self.char_lookup[char])
                except KeyError:
                    x.append(self.filler)
            X.append(x)

        new_X = np.array(X, dtype='float32')

        new_inputs = {'char_input': new_X}

        preds = self.model.predict(new_inputs,
                           batch_size = self.batch_size,
                           verbose=0)
        preds = utils.pred_to_classes(preds)

        syllabified = list([utils.stringify(o, p) for o, p in zip(data, preds)])
        if outp:
            with open(outp, 'w') as f:
                for s in syllabified:
                    f.write(s + '\n')
        else:
            return syllabified

    def vectorize(self, data):
        """
        * Will vectorize a list of syllabified words (`data`)
        * Returns np.arrays:
            - `X` the token representation of shape
              (nb_words, max_token_len + 2, nb_dims)
            - `Y` the label representation of shape
              (nb_words, max_token_len + 2, 3)
        """
        tokens, segmentations = [], []
        for token in data:
            chars, labels = [], []
            for idx, char in enumerate(token):
                if char != '-':
                    chars.append(char)
                else:
                    continue
                if idx == 0:
                    labels.append(0)
                else:
                    if token[idx - 1] == '-':
                        labels.append(1) # beginning of syllable
                    else:
                        labels.append(0)
            tokens.append(chars)
            segmentations.append(labels)

        # uniformize len of the outputs:
        outputs = []
        for segm in segmentations:
            segm = [2] + segm + [2]
            while len(segm) < (self.max_input_len + 2):
                segm.append(2)
            outputs.append(np_utils.to_categorical(segm, nb_classes=3))
        Y = np.array(outputs, dtype='int8')
        print('output shape:', Y.shape) 

        X = []
        for token in tokens:
            token = ['%'] + token + ['@'] # add beginning
            while len(token) < (self.max_input_len + 2):
                token.append('PAD')
            x = []
            for char in token:
                try:
                    x.append(self.char_lookup[char])
                except KeyError:
                    x.append(self.filler)
            X.append(x)

        X = np.array(X, dtype='float32')
        print('input shape:', X.shape)

        return X, Y

    def evaluate(self, goldp = None, silverp = None,
                       gold_data = None, silver_data = None,
                       print_score = True):
        """
        * Compares two syllabified lists in string format
          (e.g. ser-uaes):
            gold = ground truth
            silver = as predicted by system
        * Both lists can be passed as lists (`gold_data`,
          `silver_data`) or can be loaded from files 
          (`goldp`, `silverp`).
        * Will return the token-level accuracy and hyphenation
          accuracy of the silver predictions (will print these
          if `print_score` is True).

        """
        if goldp:
            gold_data = utils.load_data(goldp)
        if silverp:
            silver_data = utils.load_data(silverp)

        _, gold_Y = self.vectorize(gold_data)
        _, silver_Y = self.vectorize(silver_data)

        token_acc, hyphen_acc = utils.metrics(utils.pred_to_classes(gold_Y),
                                        utils.pred_to_classes(silver_Y))

        if print_score:
            print('\t- evaluation scores:')
            print('\t\t + token acc:', round(token_acc, 2))
            print('\t\t + hyphen acc:', round(hyphen_acc, 2))

        return token_acc, hyphen_acc


