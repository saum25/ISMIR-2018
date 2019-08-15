# ISMIR-2018

This repository contains the source code for reproducing the results of our ISMIR paper, "Understanding a Deep Machine Listening Model Through Feature Inversion". Key points about this repository are highlighted below.

1. Code to invert a feature at any later of SVDNet (SVD model from Jan Schlueter et al., ISMIR 2015).

2. Inputs to the code for performing feature inversion

	a) classifier model - jamendo_augment.npz - The code uses this to extract features from inputs. For example, the 64-dimensional vector from fc8 layer of SVDNet.

	b) inverter model - the "models" directory has pre-trained feature inverters for each layer of SVDNet. For example, jamendo_augment_gen_fc8_a26.npz is the feature inverter for the fc8 layer.

	c) dataset_directory - to read input audio

	d) results_directory - to save output plots

	e) layer - to invert features from this layer

	f) n_conv_layers, n_conv_filters - these two arguments need to be provided for inverting conv5 and other lower layers.

	g) a mean and std deviation file - jamendo_meanstd.npz

3. To invert a feature from an input, the current code works as mentioned below. Depending on the need it may be changed.

	a) create a list of mel spectrograms, one per audio file. e.g., Jamendo test dataset has 16 audio files, so the list of mel spects will have 16 elements.

	b) Then for each audio file, select an excerpt whose features are to be inverted. Currently, an excerpt per audio file between 10 sec - 20 sec of audio selected. We can change it depending on the usecase.

	c) The number of inputs to all the lasagne functions must be of batch size = 32 size. This happens due to the way the inverters are trained. So, although we feed 32 excerpts, we only care for the first excerpt in a file when we plot the results.

4. commands to invert features from each layer are mentioned below.


----- fc8 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_fc8_a26.npz ../Deep_inversion ./results --layer 'fc8'


----- fc7 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_fc7_a3.npz ../Deep_inversion ./results --layer 'fc7'


---- mp6 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_mp6_a3.npz ../Deep_inversion ./results --layer 'mp6'

----- conv5 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_conv5_a7.npz ../Deep_inversion ./results --layer 'conv5' --n_conv_layers 3 --n_conv_filters 64


----- conv4 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_conv4_a25.npz ../Deep_inversion ./results --layer 'conv4' --n_conv_layers 3 --n_conv_filters 128


---- mp3 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_mp3_a16.npz ../Deep_inversion ./results --layer 'mp3' --n_conv_layers 4 --n_conv_filters 32


----- conv2 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_conv2_a15.npz ../Deep_inversion ./results --layer 'conv2' --n_conv_layers 3 --n_conv_filters 32


---- conv1 inverter ----------

python invert.py jamendo_augment.npz models/jamendo_augment_gen_conv1_a1.npz ../Deep_inversion ./results --layer 'conv1' --n_conv_layers 1 --n_conv_filters 64

5. In Fig. 6 from the paper, in order to reproduce the results for the A and B examples, set time_index=1.4285 and 33. 


