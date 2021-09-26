import torch
import numpy as np
import os
import pickle
from module.odefunc import ODEfunc, ODEnet
from module.normalization import MovingBatchNorm1d
from module.cnf import CNF, SequentialFlow
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


from stylegan2_pytorch.model import Generator
import torch

class StyleGAN2:
    def __init__(self):
        # self.opt = opt
        if os.path.exists("/media/morzh/ext4_volume/work/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt"):
            print("Found local StyleGan2!")
            network_pkl = "/media/morzh/ext4_volume/work/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt"
        else:
            network_pkl = self.opt.network_pkl

        self.latent = 512
        self.n_mlp = 8
        self.size = 1024
        self.channel_multiplier = 2
        self.truncation = 1
        self.truncation_mean = 4096
        self.rand_sample = 1
        self.device = torch.device("cpu")
        self.g_ema = Generator(self.size, self.latent, self.n_mlp, channel_multiplier=self.channel_multiplier).to(self.device)
        self.g_ema.load_state_dict(torch.load(network_pkl)["g_ema"])
        self.mean_latent = self.g_ema.mean_latent(self.truncation_mean)

    # def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
    #     pass
    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        # print('generate_im_from_z_space', type(z), z.shape)
        images = self.g_ema(torch.from_numpy(z), truncation=truncation_psi, truncation_latent=self.mean_latent, input_is_latent=False)
        return images

    def generate_im_from_w_space(self, w, truncation=0.75):
        # print('generate_im_from_w_space', type(w), w.shape)
        # print('mean latent', self.mean_latent.shape)
        # w = torch.from_numpy(w)
        # print('generate_im_from_w_space', type(w), w.shape)
        images, _ = self.g_ema([w], truncation=truncation, truncation_latent=self.mean_latent, input_is_latent=True)
        # print('images shape is:', len(images))

        img = images[0].permute(1, 2, 0)
        img = (img + 1.0) / 2.0
        img *= 255.0
        img = img.detach().cpu().numpy().astype(np.uint8).copy()

        # print('image shape', img.shape)
        # print('type', type(img))
        # print('dtype', img.dtype)

        return img


class StyleFlow:
    def __init__(self, model_filepath, all_w_filepath, all_tsne_filepath, all_attr_filepath, all_lights_filepath):
        self.square_size = 50
        self.attr_interval = 80
        self.attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]
        self.attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
        self.lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

        self.min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0}
        self.max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}

        self.interval_dic = {'Gender': self.attr_interval, 'Glasses': self.attr_interval, 'Yaw': self.attr_interval, 'Pitch': self.attr_interval,
                             'Baldness': self.attr_interval, 'Beard': self.attr_interval, 'Age': self.attr_interval, 'Expression': self.attr_interval}

        self.set_values_dic = {i: int(self.interval_dic[i]/2) for i in self.interval_dic}
        self.gap_dic = {i: self.max_dic[i] - self.min_dic[i] for i in self.max_dic}
        self.dimensions_str = '512-512-512-512-512'
        self.input_dim = 512
        self.zdim = 17
        self.num_blocks = 1
        self.zero_padding = torch.zeros(1, 18, 1).cpu()

        self.keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362, 369, 462, 460, 478, 551,
                             583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301, 599]

        self.fws = None
        self.q_array = None
        self.w_current = np.ndarray
        self.attr_current = None
        self.light_current = None
        self.rev = None
        self.attr_current_list = []
        self.light_current_list = []
        self.array_source = None
        self.array_light = None
        self.pre_lighting_distance = None

        self.all_w = pickle.load(open(all_w_filepath, "rb"))['Latent']
        self.all_tsne = np.load(all_tsne_filepath)
        self.all_attr = np.load(all_attr_filepath)
        self.all_lights2 = np.load(all_lights_filepath)
        self.all_lights = self.all_lights2
        self.final_array_source = None
        self.final_array_target = None

        light0 = torch.from_numpy(self.all_lights2[8]).type(torch.FloatTensor).cpu()
        light1 = torch.from_numpy(self.all_lights2[33]).type(torch.FloatTensor).cpu()
        light2 = torch.from_numpy(self.all_lights2[641]).type(torch.FloatTensor).cpu()
        light3 = torch.from_numpy(self.all_lights2[547]).type(torch.FloatTensor).cpu()
        light4 = torch.from_numpy(self.all_lights2[28]).type(torch.FloatTensor).cpu()
        light5 = torch.from_numpy(self.all_lights2[34]).type(torch.FloatTensor).cpu()

        self.pre_lighting = [light0, light1, light2, light3, light4, light5]

        dimensions = tuple(map(int, self.dimensions_str.split("-")))
        self.build_model(self.input_dim, dimensions, self.zdim, self.num_blocks, True)
        self.model.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')))
        self.model.eval()

    def build_model(self, input_dim, hidden_dims, context_dim, num_blocks, conditional):
        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type='concatsquash',
                nonlinearity='tanh',
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
            )
            cnf = CNF(
                odefunc=odefunc,
                T=1.0,
                train_T=True,
                conditional=conditional,
                solver='dopri5',
                use_adjoint=True,
                atol=1e-5,
                rtol=1e-5,
            )
            return cnf

        chain = [build_cnf() for _ in range(num_blocks)]
        bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False) for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False)]
        for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
        chain = bn_chain
        self.model = SequentialFlow(chain)

    def invert_slide_to_real(self, name, slide_value):
        return float(slide_value / self.interval_dic[name] * (self.gap_dic[name]) + self.min_dic[name])

    def light_transfer_real_to_slide(self, name, real_value):
        return int((real_value - self.light_min_dic[name]) / (self.light_gap_dic[name]) * self.light_interval_dic[name])

    def load_sample(self, index):
        # print('update_GT_scene_image')
        idx = self.keep_indexes[index]
        self.w_current = self.all_w[idx].copy()
        self.attr_current = self.all_attr[idx].copy()
        self.light_current = self.all_lights[idx].copy()

        self.attr_current_list = [self.attr_current[i][0] for i in range(len(self.attr_order))]
        self.light_current_list = [0 for i in range(len(self.lighting_order))]
        '''
        for i, j in enumerate(self.attr_order):
            self.slider_list[i].setValue(self.transfer_real_to_slide(j, self.attr_current_list[i]))

        for i, j in enumerate(self.lighting_order):
            self.lighting_slider_list[i].setValue(0)
        '''
        ################################  calculate attributes array first, then change the values of attributes
        self.q_array = torch.from_numpy(self.w_current).cpu().clone().detach()
        self.array_source = torch.from_numpy(self.attr_current).type(torch.FloatTensor).cpu()
        self.array_light = torch.from_numpy(self.light_current).type(torch.FloatTensor).cpu()
        self.pre_lighting_distance = [self.pre_lighting[i] - self.array_light for i in range(len(self.lighting_order))]

        self.final_array_source = torch.cat([self.array_light, self.array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        self.final_array_target = torch.cat([self.array_light, self.array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        # print(self.q_array.shape, self.final_array_source.shape, self.zero_padding.shape)
        self.fws = self.model(self.q_array, self.final_array_source, self.zero_padding)

    def augment_attributes(self, attr_index, raw_slide_value):
        real_value = self.invert_slide_to_real(self.attr_order[attr_index], raw_slide_value)
        attr_change = real_value - self.attr_current_list[attr_index]
        attr_final = self.attr_degree_list[attr_index] * attr_change + self.attr_current_list[attr_index]

        self.final_array_target[0, attr_index + 9, 0, 0] = attr_final
        self.rev = self.model(self.fws[0], self.final_array_target, self.zero_padding, True)
        '''
        np.save('/home/morzh/work/StyleFlow/face_project_app/test_data/face_project_fws', self.fws[0].detach().numpy())
        np.save('/home/morzh/work/StyleFlow/face_project_app/test_data/face_project_q_array', self.q_array.detach().numpy())
        with open('/home/morzh/work/StyleFlow/face_project_app/test_data/face_project_rev.pickle', 'wb') as f:
            pickle.dump(self.rev, f)

        self.fws = (torch.from_numpy(np.load('/home/morzh/work/fws.npy')), self.fws[1])
        self.q_array = torch.from_numpy(np.load('/home/morzh/work/q_array.npy'))
        with open('/home/morzh/work/rev.pickle', 'rb') as f:
            self.rev = pickle.load(f)
        '''
        if attr_index == 0:
            self.rev[0][0][8:] = self.q_array[0][8:]

        elif attr_index == 1:
            self.rev[0][0][:2] = self.q_array[0][:2]
            self.rev[0][0][4:] = self.q_array[0][4:]

        elif attr_index == 2:
            self.rev[0][0][4:] = self.q_array[0][4:]

        elif attr_index == 3:
            self.rev[0][0][4:] = self.q_array[0][4:]

        elif attr_index == 4:
            self.rev[0][0][6:] = self.q_array[0][6:]

        elif attr_index == 5:
            self.rev[0][0][:5] = self.q_array[0][:5]
            self.rev[0][0][10:] = self.q_array[0][10:]

        elif attr_index == 6:
            self.rev[0][0][0:4] = self.q_array[0][0:4]
            self.rev[0][0][8:] = self.q_array[0][8:]

        elif attr_index == 7:
            self.rev[0][0][:4] = self.q_array[0][:4]
            self.rev[0][0][6:] = self.q_array[0][6:]

        self.w_current = self.rev[0].detach().cpu().numpy()
        self.q_array = torch.from_numpy(self.w_current).cpu().clone().detach()
        self.fws = self.model(self.q_array, self.final_array_target, self.zero_padding)

        print('real_value: ', real_value)
        print('attr_change: ', attr_change)
        print('attr_final: ', attr_final)
        print('attr_current_list:', self.attr_current_list)
        print('final_array_target:', torch.flatten(self.final_array_target))
        print('final_array_source:', torch.flatten(self.final_array_source))

    def save_state(self, path, w_filename, attr_filename, light_filename):
        np.save(os.path.join(path, w_filename), self.w_current)
        np.save(os.path.join(path, attr_filename), self.attr_current)
        np.save(os.path.join(path, light_filename), self.light_current)

    def load_state(self, path, w_filename, attr_filename, light_filename):
        self.w_current = np.load(os.path.join(path, w_filename))
        self.attr_current = np.save(os.path.join(path, attr_filename))
        self.light_current = np.save(os.path.join(path, light_filename))



stylegan = StyleGAN2()
styleflow = StyleFlow('../flow_weight/modellarge10k.pt', '../data/sg2latents.pickle', '../data/TSNE.npy', '../data/attributes.npy', '../data/light.npy')

styleflow.load_sample(16)
# styleflow.save_state('/home/morzh/work/StyleFlow/face_project_app/test_data', 'face_project_w', 'face_project_attr', 'face_project_light')
plt.figure(figsize=(11, 10))
plt.imshow(stylegan.generate_im_from_w_space(torch.from_numpy(styleflow.w_current)))
plt.tight_layout()
plt.show()

styleflow.augment_attributes(5, 80)
styleflow.augment_attributes(4, 80)
# np.save('/home/morzh/work/StyleFlow/face_project_app/test_data/face_project_fws', styleflow.fws[0].detach().numpy())

plt.figure(figsize=(11, 10))
plt.imshow(stylegan.generate_im_from_w_space(torch.from_numpy(styleflow.w_current)))
plt.tight_layout()
plt.show()

