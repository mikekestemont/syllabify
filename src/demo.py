#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Syllabify import Syllabifier
import utils

def main():
    """
    Demo
    """

    syllabifier = Syllabifier(model_dir = 'model_s',
                              load = True)

    info = '\n' * 10
    info += "######################################################\n"
    info += "##### Syllabification Demo for Middle Dutch #########\n"
    info += "#####################################################\n\n"
    print(info)

    phrase = 'Enter a Middle Dutch word (or type QUIT to stop): '
    word = ''
    
    while word != 'QUIT':
        word = input(phrase)
        segmented = syllabifier.syllabify(data=[word])[0]
        print('Segmentation proposed: '+ segmented)

if __name__ == '__main__':
    main()