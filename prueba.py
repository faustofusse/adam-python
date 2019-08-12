from deeppavlov import build_model, configs
import pickle

# print (configs.squad.squad)
path = './DeepPavlov/deeppavlov/configs/squad/squad.json'
pathMulti = './DeepPavlov/deeppavlov/configs/squad/squad_ru_bert_infer.json'

model = build_model(pathMulti, download=True)
model(['Hoss cocino una torta de manzana ayer a la noche.'], ['Que hizo Hoss?'])
# https://github.com/deepmipt/DeepPavlov
# python -m deeppavlov interact DeepPavlov/deeppavlov/configs/squad/squad_bert.json
