from preprocessing_01 import preprocessing_demo, get_data
import numpy as np

data_dir = '../images/'
preprocessing_demo(data_dir,4,3)  

# # get_data(data_dir, scale):
normal_lr, normal_hr = get_data(data_dir, 4)
print(len(normal_hr))

# lr 0 to 1
a = np.array(normal_lr)
print('lr',np.min(a),np.max(a))

# -1 to 1 
b = np.array(normal_hr)
print('hr',np.min(b),np.max(b))

