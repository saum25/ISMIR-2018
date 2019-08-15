

import argparse


def argument_parser():
	'''
	parses the command line arguments
	'''
	parser = argparse.ArgumentParser(description='generates a mel spectrogram from an input feature')
	parser.add_argument('classifier', action='store', help='pre-trained classifier (.npz format)')
	parser.add_argument('inverter', action='store', help='pre-trained feature inverter (.npz format)')
	parser.add_argument('dataset_path', action='store', help='dataset path')
	parser.add_argument('results_dir', action='store', help='path to save inversion plots')
	parser.add_argument('--dataset', default='jamendo', help='dataset name')
	parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='store spectra in the given directory (disabled by default).')
	parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
	parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.')
	parser.add_argument('--featloss', default=False, action='store_true', help='If given, calculate feature space loss.')
	parser.add_argument('--layer', default='fc8', help='invert features from this classifier layer')
	parser.add_argument('--n_conv_layers', default=1, type=int, help='number of 3x3 conv layers to be added before upconv layers')
	parser.add_argument('--n_conv_filters', default=32, type=int, help='number of filters per conv layer in the upconvolutional architecture for Conv layer inversion')	
	return parser

def normalise(x):
	'''
	Normalise a vector/ matrix, in range 0 - 1
    '''
	return((x-x.min())/(x.max()-x.min()))