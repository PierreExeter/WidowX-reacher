conda env create -f environment/environment.yml

cd gym_replab
pip install -e .
cd ..

cd rlkit
pip install -e .
cd ..


cd viskit
pip install -e .
cd ..

python examples/td3.py

