from deeppavlov import build_model, configs
import pickle

# print (configs.squad.squad)
path = './DeepPavlov/deeppavlov/configs/squad/squad.json'
path2 = './models/models/squad_model/model'

model = build_model(path, download=True);
model(['Hoss cocino una torta de manzana ayer a la noche.'], ['Que hizo Hoss?'])

# python -m deeppavlov interact DeepPavlov/deeppavlov/configs/squad/squad_bert.json
