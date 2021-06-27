import pickle
import pandas as pd
import os


path = '/media/morzh/ext4_volume/data/datasets/dataset_styleflow'

filename_all_att = 'all_att.pickle'
filename_all_latents = 'all_latents.pickle'
filename_all_light10k = 'all_light10k.pickle'


dict_all_att = pd.read_pickle(os.path.join(path, filename_all_att))
dict_all_latents = pd.read_pickle(os.path.join(path, filename_all_latents))
dict_all_light10k = pd.read_pickle(os.path.join(path, filename_all_light10k))

print('all_att.pickle dataset samples is', len(dict_all_att['Attribute'][0]))
print('all_latents.pickle dataset samples is', dict_all_latents['Latent'].shape[0])
print('all_light10k.pickle dataset samples is', len(dict_all_light10k['Light']))


