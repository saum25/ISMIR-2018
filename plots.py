'''
Created on 9 Dec 2017

@author: Saumitra
'''

import matplotlib.pyplot as plt
import librosa.display as disp

def plot_figures(input_excerpt, inv, pred, file_id, excerpt_id, results_path, layer):

    fs = 9
    plt.figure(figsize=(4,4))
    
    plt.subplot(2, 1, 1)
    disp.specshow(input_excerpt.T, y_axis='mel', hop_length= 315, x_axis='off', fmin=27.5, fmax=8000, cmap = 'coolwarm')
    plt.title('input mel spectrogram', fontsize = fs)
    plt.ylabel('Hz', fontsize = fs, labelpad = 1)
    plt.yticks(fontsize = fs)
    
    plt.subplot(2, 1, 2)
    plt.title(layer + ' '+ 'inversion', fontsize = fs)
    disp.specshow(inv.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
    plt.xlabel('Time', fontsize = fs, labelpad = 1)
    plt.ylabel('Hz', fontsize = fs, labelpad = 1)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
       
    plt.tight_layout()
    plt.savefig(results_path + '/'+ 'plot'+ '_fileid_'+ str(file_id) + '_excerptid_' + str(excerpt_id) + '_pred_'+ "%.2f"  %pred +'.pdf', dpi = 300)
