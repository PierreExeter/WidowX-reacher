import pickle
import joblib


path = 'rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/'

input_file = path+'params.pkl'
output_file = path+'cleaned_params.pkl'

data = joblib.load(input_file)

del data['env']

print(data)

with open(output_file, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



