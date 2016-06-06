# Syllabification for Middle Dutch

## Introduction

The repository contains a simple, yet highly efficient implementation of a syllabification engine, originally developped for Middle Dutch words. The code allows to train models on annotated training data, save them and apply them to new, unseen words. Our model architecture is based on a fairly straightforward character-level recurrent neural network to produce syllable segmentations on the basis of the output of a stack of Long-Short Term Memory layers (http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735#.V1UuiJN95E4)[[original paper]]. Even on a normal CPU, 30 epochs of the model can be run in under an hour for 20,000 training items.


## Dependencies

The code uses Python 3.4+ and has the following major dependencies (preferably bleeding edge versions from Anaconda's Python distribution or Github):
- [numpy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [keras](http://keras.io/), which we used with [Theano](http://deeplearning.net/software/theano/) as backend.


## Details

For more information about the task, consult this paper: Gosse Bouma and Ben Hermans, Syllabification of Middle Dutch in Francesco Mambrini, Marco Passarotti, and Corline Sporleder, Proceedings of the Second Workshop on Annotation of Corpora for Research in the Humanities, pp. 27-39 [https://www.let.rug.nl/~gosse/papers/hyphenating_crm.pdf]([pdf]). This repository has been in the framework of the FWO-funded PhD project [https://www.uantwerpen.be/en/staff/wouter-haverals/research/](The Measure of Middle Dutch: Rhythm and Prosody Reconstruction for Middle Dutch Literature, A Data-Driven Approach) carried out by Wouter Haverals at the University of Antwerp (supervisors: F. Willaert and M. Kestemont). For further information about this repository or project, contact wouter.haverals@uantwerp.be.

