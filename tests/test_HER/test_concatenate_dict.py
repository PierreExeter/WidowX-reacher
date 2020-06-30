from collections import OrderedDict
import numpy as np


KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']

obs_dict = OrderedDict([
    ('achieved_goal', np.array([[5.593112e-03, 8.580400e-05, 3.641368e-01]], dtype=np.float32)), 
    ('desired_goal', np.array([[0.01366778, 0.05594923, 0.33835924]], dtype=np.float32)), 
    ('observation', np.array([[ 5.5931122e-03,  8.5804000e-05,  3.6413679e-01,  1.5339808e-02, -1.2931458e+00,  1.0109711e+00, -1.3537670e+00, -7.1585767e-02, 0.0000000e+00]], dtype=np.float32))
    ])




obs_dict1 = OrderedDict([
    ('achieved_goal', np.array([-0.00758535, -0.00137682,  0.3674703 ], dtype=np.float32)), 
    ('desired_goal', np.array([ 0.01256729, -0.01984975,  0.34396625], dtype=np.float32)), 
    ('observation', np.array([-0.00758535, -0.00137682,  0.3674703 ,  0.17955539, -1.3640159 , 0.98847556, -1.2892197 ,  0.07993339,  0.        ], dtype=np.float32))
    ])




# print(obs_dict)
# print(obs_dict1)


# print(len(obs_dict1[KEY_ORDER[0]].shape))



# for key in KEY_ORDER:
#     obs_dict[key] = obs_dict[key][0]
    
    # concat = np.concatenate([obs_dict[key] for key in KEY_ORDER])
    # print(concat)

# print(obs_dict)


def convert_dict_to_obs(obs_dict):

    if len(obs_dict[KEY_ORDER[0]].shape) == 2:
        for key in KEY_ORDER:
            obs_dict[key] = obs_dict[key][0]

    return np.concatenate([obs_dict[key] for key in KEY_ORDER])

# # print("-----------")

# # print([obs_dict[key] for key in KEY_ORDER])
# # print([obs_dict1[key] for key in KEY_ORDER])

print(convert_dict_to_obs(obs_dict))
print(convert_dict_to_obs(obs_dict1))
