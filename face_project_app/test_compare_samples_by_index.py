import numpy as np
import os

path_style_flow_gui = '/home/morzh/work'
path_style_flow_face_project = '/home/morzh/work/StyleFlow/face_project_app/test_data'

w_style_flow_gui = np.load(os.path.join(path_style_flow_gui, 'w_current.npy'))
attr_style_flow_gui = np.load(os.path.join(path_style_flow_gui, 'attr_current.npy'))
light_style_flow_gui = np.load(os.path.join(path_style_flow_gui, 'light_current.npy'))

w_style_flow_face_project = np.load(os.path.join(path_style_flow_face_project, 'face_project_w.npy'))
attr_style_flow_face_project = np.load(os.path.join(path_style_flow_face_project, 'face_project_attr.npy'))
light_style_flow_face_project = np.load(os.path.join(path_style_flow_face_project, 'face_project_light.npy'))

print(np.all(w_style_flow_gui == w_style_flow_face_project))
print(np.all(attr_style_flow_gui == attr_style_flow_face_project))
print(np.all(light_style_flow_gui == light_style_flow_face_project))
