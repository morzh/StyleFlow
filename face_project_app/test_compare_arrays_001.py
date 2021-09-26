import numpy as np
import os
import pickle

path_style_flow_gui = '/home/morzh/work'
path_style_flow_face_project = '/home/morzh/work/StyleFlow/face_project_app/test_data'

fws_style_flow_gui = np.load(os.path.join(path_style_flow_gui, 'fws.npy'))
q_array_style_flow_gui = np.load(os.path.join(path_style_flow_gui, 'q_array.npy'))
with open(os.path.join(path_style_flow_gui, 'rev.pickle'), 'rb') as f:
    rev_style_flow_gui = pickle.load(f)

fws_style_flow_face_project = np.load(os.path.join(path_style_flow_face_project, 'face_project_w.npy'))
q_array_style_flow_face_project = np.load(os.path.join(path_style_flow_face_project, 'face_project_light.npy'))
with open(os.path.join(path_style_flow_face_project, 'face_project_rev.pickle'), 'rb') as f:
    rev_style_flow_face_project = pickle.load(f)

sse_fws = np.sum(np.abs(fws_style_flow_gui - fws_style_flow_face_project))
sse_q_array = np.sum(np.abs(q_array_style_flow_gui - q_array_style_flow_face_project))
sse_rev = tuple(x - y for x, y in zip(rev_style_flow_gui, rev_style_flow_face_project))

print(sse_fws)
print(sse_q_array)
print(sse_rev)
