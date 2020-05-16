# install
conda env create -f environment.yml
conda activate rlkit

cd gym_replab
pip install -e .
cd ..

cd rlkit
pip install -e .
cd ..


cd viskit
pip install -e .
cd ..

# train
python train_scripts/td3.py


# plot training stats
python viskit/viskit/frontend.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_20000--s-0/


# evaluate a trained policy
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_26_0000--s-0/params.pkl

python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_25_35_0000--s-0/params.pkl

python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/params.pkl

python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/cleaned_params.pkl


# visualise a trained policy
python enjoy_scripts/save_clean_pickle.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_26_0000--s-0/params.pkl
python enjoy_scripts/simple_sim_policy.py
