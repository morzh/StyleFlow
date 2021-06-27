import numpy as np
import torch
from module.cnf import *
from module.flow import *
import pickle


light_degree = 1
light_min_dic = {'light': 0}
light_max_dic = {'light': light_degree}
light_interval = 80
light_interval_dic = {'light': light_interval}
light_set_values_dic = {i: 0 for i in light_interval_dic}
light_gap_dic = {i: light_max_dic[i] - light_min_dic[i] for i in light_max_dic}

min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': 0, 'Pitch': 0, 'Baldness': 0, 'Beard': 0, 'Age': 0, 'Expression': 0}
max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 1, 'Pitch': 1, 'Baldness': 1, 'Beard': 1, 'Age': 1, 'Expression': 1}

attr_interval = 80
interval_dic = {'Gender': attr_interval, 'Glasses': attr_interval, 'Yaw': attr_interval, 'Pitch': attr_interval,
                'Baldness': attr_interval, 'Beard': attr_interval, 'Age': attr_interval, 'Expression': attr_interval}
# set_values_dic = {i: int(interval_dic[i]/2) for i in interval_dic}
gap_dic = {i: max_dic[i] - min_dic[i] for i in max_dic}


def pre_computing_distance(pre_lighting, array_light, lighting_order, R_attr_current_list, attr_current_list, attr_order):
    pre_lighting_distance = [pre_lighting[i] - array_light for i in range(len(lighting_order))]
    pre_attr_distance = [R_attr_current_list[i] - attr_current_list[i] for i in range(len(attr_order))]
    return pre_lighting_distance, pre_attr_distance


def invert_slide_to_real(name, slide_value):
    return float(slide_value /interval_dic[name] * (gap_dic[name]) + min_dic[name])

def transfer_real_to_slide(name, real_value):
    return int((real_value - min_dic[name]) / (gap_dic[name]) * interval_dic[name])

# self.opt = opt
# self.model = Build_model(self.opt)
# self.w_avg = self.model.Gs.get_var('dlatent_avg')
#

keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301, 599]
keep_indexes = np.array(keep_indexes).astype(np.int)

prior = cnf(512, '512-512-512-512-512', 17, 1)
prior.load_state_dict(torch.load('/media/morzh/ext4_volume/work/StyleFlow/flow_weight/modellarge10k.pt', map_location=torch.device('cpu')))
prior.eval()

attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
# lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']
lighting_order = ['light']
attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.0]


raw_w = pickle.load(open("/media/morzh/ext4_volume/work/StyleFlow/data/sg2latents.pickle", "rb"))
raw_TSNE = np.load('/media/morzh/ext4_volume/work/StyleFlow/data/TSNE.npy')
raw_attr = np.load('/media/morzh/ext4_volume/work/StyleFlow/data/attributes.npy')
raw_lights = np.load('/media/morzh/ext4_volume/work/StyleFlow/data/light.npy')


all_w = np.array(raw_w['Latent'])[keep_indexes]
all_attr = raw_attr[keep_indexes]
all_lights = raw_lights[keep_indexes]

import torch
from stylegan2_pytorch.model import Generator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

latent = 512
n_mlp = 8
size = 1024
channel_multiplier = 2
truncation = 1
truncation_mean = 4096
rand_sample = 1

ckpt = '/media/morzh/ext4_volume/work/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt'

device = torch.device("cpu")
g_ema = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
checkpoint = torch.load(ckpt)
g_ema.load_state_dict(checkpoint["g_ema"])
mean_latent = g_ema.mean_latent(truncation_mean)
zero_padding = torch.zeros(1, 18, 1).cpu()

for index in range(1000):
    attr_current = all_attr[index]
    light_current = all_lights[index]
    w_current = torch.from_numpy(all_w[index])

    attr_current_list = [attr_current[i][0] for i in range(len(attr_order))]
    light_current_list = [0 for i in range(len(lighting_order))]

    array_source = torch.from_numpy(attr_current).type(torch.FloatTensor).cpu()
    array_light = torch.from_numpy(light_current).type(torch.FloatTensor).cpu()
    '''
    slider_list = []
    lighting_slider_list = []

    for i, j in enumerate(attr_order):
        slider_list[i].setValue(transfer_real_to_slide(j, attr_current_list[i]))

    for i, j in enumerate(lighting_order):
        lighting_slider_list[i].setValue(0)
    '''

    final_array_source = torch.cat([array_light, array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
    final_array_target = torch.cat([array_light, array_source.unsqueeze(0).unsqueeze(-1)], dim=1)

    R_attr_current = all_attr[index].copy()
    R_attr_current_list = [R_attr_current[i][0] for i in range(len(attr_order))]
    R_light_current = all_lights[index].copy()
    pre_lighting = [torch.from_numpy(R_light_current).type(torch.FloatTensor).cpu()]

    print('gender:', attr_current[0], 'glasses:', attr_current[1], 'yaw:', attr_current[2], 'pitch:', attr_current[3],
          'bald:', attr_current[4], 'beard:', attr_current[5], 'age:', attr_current[6], 'expression:', attr_current[7])

    pre_lighting_distance, pre_attr_distance = pre_computing_distance(pre_lighting, array_light, lighting_order, R_attr_current_list, attr_current_list, attr_order)

    attr_index = 6
    real_value = invert_slide_to_real(attr_order[attr_index], 50.0)
    attr_change = real_value - attr_current_list[attr_index]
    attr_final = attr_degree_list[attr_index] * attr_change + attr_current_list[attr_index]

    final_array_target[0, attr_index + 9, 0, 0] = attr_final
    zero_padding = torch.zeros(1, 18, 1).cpu()


    print('generating image 1')
    print('w_current.shape', w_current.shape)
    sample, _ = g_ema([w_current], truncation=truncation, truncation_latent=mean_latent, input_is_latent=True)
    # q_array = w_current.cpu().clone().detach()
    # w_new = prior(w_current, final_array_target, zero_padding)
    fws = prior(w_current, final_array_target, zero_padding, True)
    # rev = prior(fws[0], final_array_target, zero_padding, True)
    eee = fws[0].detach().cpu()
    q_array = w_current.cpu().clone().detach()
    
    if attr_index == 0:
        eee[0][8:] = q_array[0][8:]

    elif attr_index == 1:
        eee[0][:2] = q_array[0][:2]
        eee[0][4:] = q_array[0][4:]

    elif attr_index == 2:
        eee[0][4:] = q_array[0][4:]

    elif attr_index == 3:
        eee[0][4:] = q_array[0][4:]

    elif attr_index == 4:
        eee[0][6:] = q_array[0][6:]

    elif attr_index == 5:
        eee[0][:5] = q_array[0][:5]
        eee[0][10:] = q_array[0][10:]

    elif attr_index == 6:
        eee[0][0:4] = q_array[0][0:4]
        eee[0][8:] = q_array[0][8:]

    elif attr_index == 7:
        eee[0][:4] = q_array[0][:4]
        eee[0][6:] = q_array[0][6:]

    w_current = eee.detach().cpu()
    

    # fws = prior(q_array, final_array_target, zero_padding)
    # sample_new, _ = g_ema([w_current], truncation=truncation, truncation_latent=mean_latent, input_is_latent=True)

    sample_new, _ = g_ema([eee], truncation=truncation, truncation_latent=mean_latent, input_is_latent=True)

    img = sample[0].permute(1, 2, 0)
    img_new = sample_new[0].permute(1, 2, 0)
    img = (img + 1.0) / 2.0
    img_new = (img_new + 1.0) / 2.0

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img.detach().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(img_new.detach().numpy())
    plt.tight_layout()
    plt.show()
    print('===========================================')

