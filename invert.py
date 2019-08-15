'''
Created on 14 Aug 2019

Code to generate a mel-spectrogram from an input
feature at a layer l of CNN SVD. For more information
refer to ISMIR 2018 paper on feature inversion.

http://ismir2018.ircam.fr/doc/pdfs/272_Paper.pdf

@author: Saumitra
'''

import io
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
floatX = theano.config.floatX
import utils
from progress import progress
from simplecache import cached
import audio
import model
import upconv
import plots

def main():
    # parse the command line arguments
    parser = utils.argument_parser()
    args = parser.parse_args()
    
    print("-------------------------------")
    print("classifier:%s" %args.classifier)
    print("inverter:%s" %args.inverter)
    print("dataset_path:%s" %args.dataset_path)
    print("dataset name:%s" %args.dataset)
    print("results path:%s" %args.results_dir)
    print("inverting from: %s" %args.layer)
    print("-------------------------------")
    
    # default parameters 
    sample_rate = 22050
    frame_len = 1024
    fps = 70
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    blocklen = 115
    batchsize = 32
    start_offset = 10 # secs
    end_offset = 20 # secs    
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    
    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__), args.dataset_path, 'datasets', args.dataset)
    
    # load filelist
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # compute spectra
    print("Computing%s spectra..." %
          (" or loading" if args.cache_spectra else ""))
    
    spects = [] # list of tuples, where each tuple has magnitude and phase information for one audio file
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
        
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank), 1e-7)) for spect in spects)
    
    # load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    print("Preparing training data feed...")
    # normalised mel spects, without data augmentation
    mel_spects = [(spect - mean) * istd for spect in mel_spects]
    
    # we create two theano functions
    # the first one uses pre-trained classifier to generate features and predictions
    # the second one uses pre-trained inverter to generate mel spectrograms from input features 
    
    # classifier (discriminator) model
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension, changes a 32 x 115 x 80 input to 32 x 1 x 115 x 80 input which is fed to the CNN
    
    network = model.architecture(inputs, (None, 1, blocklen, mel_bands))
    
    # load saved weights
    with np.load(args.classifier) as f:
        lasagne.layers.set_all_param_values(
                network['fc9'], [f['param%d' % i] for i in range(len(f.files))])
        
    # create output expression
    outputs_score = lasagne.layers.get_output(network[args.layer], deterministic=True)
    outputs_pred = lasagne.layers.get_output(network['fc9'], deterministic=True)

    # prepare and compile prediction function
    print("Compiling classifier function...")
    pred_fn_score = theano.function([input_var], outputs_score, allow_input_downcast= True)
    pred_fn = theano.function([input_var], outputs_pred, allow_input_downcast= True)
    
    # inverter (generator) model    
    if (args.layer == 'fc8') or (args.layer == 'fc7'):
        input_var_deconv = T.matrix('input_var_deconv')
    else:
        input_var_deconv = T.tensor4('input_var_deconv')


    # inverter (generator) model    
    if (args.layer == 'fc8'):
        gen_network = upconv.architecture_upconv_fc8(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1]))
    elif args.layer == 'fc7':
        gen_network = upconv.architecture_upconv_fc7(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1]))
    elif args.layer == 'mp6':
        gen_network = upconv.architecture_upconv_mp6(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    elif args.layer == 'conv5':
        gen_network = upconv.architecture_upconv_conv5(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    elif args.layer == 'conv4':
        gen_network = upconv.architecture_upconv_conv4(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    elif args.layer == 'mp3':
        gen_network = upconv.architecture_upconv_mp3(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    elif args.layer == 'conv2':
        gen_network = upconv.architecture_upconv_conv2(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    else:
        gen_network = upconv.architecture_upconv_conv1(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network[args.layer])[1], lasagne.layers.get_output_shape(network[args.layer])[2], lasagne.layers.get_output_shape(network[args.layer])[3]), args.n_conv_layers, args.n_conv_filters)
    
    # load saved weights
    with np.load(args.inverter) as f:
        lasagne.layers.set_all_param_values(
                gen_network, [f['param%d' % i] for i in range(len(f.files))])
    
    # create cost expression
    outputs = lasagne.layers.get_output(gen_network, deterministic=True)
    print("Compiling inverter function...")
    test_fn = theano.function([input_var_deconv], outputs, allow_input_downcast= True)
    
    # instance-based feature inversion
    # (1) pick a file from a dataset (e.g., dataset: Jamendo test) (2) select a time index to read the instance
    file_idx = np.arange(0, len(filelist))
    hop_size= sample_rate/fps # samples

    for file_instance in file_idx:
        print("<<<<Analysis for the file: %d>>>>" %(file_instance+1))
        time_idx = np.random.randint(start_offset, end_offset, 1)[0]   # provides a random integer start position between start and end offsets
        
        # generate excerpts for the selected file_idx
        # excerpts is a 3-d array of shape: num_excerpts x blocklen x mel_spects_dimensions   
        num_excerpts = len(mel_spects[file_instance]) - blocklen + 1
        print("Number of excerpts in the file :%d" %num_excerpts)
        excerpts = np.lib.stride_tricks.as_strided(mel_spects[file_instance], shape=(num_excerpts, blocklen, mel_spects[file_instance].shape[1]), strides=(mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[1]))
        
        # convert the time_idx to the excerpt index
        excerpt_idx = int(np.round((time_idx * sample_rate)/(hop_size)))
        print("Time_idx: %f secs, Excerpt_idx: %d" %(time_idx, excerpt_idx))
        if ((excerpt_idx + batchsize) > num_excerpts):
            print("------------------Number of excerpts are less for file: %d--------------------" %(file_instance+1))
            break
    
        # generating feature representations for the select excerpt.
        # CAUTION: Need to feed mini-batch to pre-trained model, so (mini_batch-1) following excerpts are also fed, but are not analysed
        # classifier can have less than minibatch data, but the inverter needs a batch of data to make prediction (comes from how the inverter was trained)
        scores = pred_fn_score(excerpts[excerpt_idx:excerpt_idx + batchsize])
        #print("Feature"),
        #print(scores[file_idx])
        
        predictions = pred_fn(excerpts[excerpt_idx:excerpt_idx + batchsize])
        #print("Prediction:%f" %(predictions[0][0]))
        
        mel_predictions = np.squeeze(test_fn(scores), axis = 1) # mel_predictions is a 3-d array of shape batch_size x blocklen x n_mels
        
        # saves plots for the input Mel spectrogram and its inverted representation
        # all plots are normalised in [0, 1] range
        plots.plot_figures(utils.normalise(excerpts[excerpt_idx]), utils.normalise(mel_predictions[0]), predictions[0][0], file_instance, excerpt_idx, args.results_dir, args.layer)
        
if __name__ == '__main__':
    main()


