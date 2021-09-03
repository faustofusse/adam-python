from pathlib import Path
from .resources.chainer import Chainer
from .resources.utils import import_packages, parse_config
from .resources.params import from_params
from .resources.infer import build_model
import json
import pickle
import os

config_path = './question_and_answer/config/squad_ru_bert_infer.json'
model_path = './question_and_answer/deeppavlov/models/saved/model1.pickle'


class QuestionAnswerModel(object):
    def __init__(self):
        self.model = build_model(Path(config_path))

    def interact(self, payload):
        model_args = payload.values()
        error_msg = None
        lengths = {len(model_arg)
                   for model_arg in model_args if model_arg is not None}

        if not lengths:
            error_msg = 'got empty request'
        elif 0 in lengths:
            error_msg = 'dot empty array as model argument'
        elif len(lengths) > 1:
            error_msg = 'got several different batch sizes'

        if error_msg is not None:
            print('Error:', error_msg)
            return [{'error': error_msg}]

        batch_size = next(iter(lengths))
        model_args = [arg or [None] * batch_size for arg in model_args]

        prediction = self.model(*model_args)
        if len(self.model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))
        return prediction

# print(interact(model, {"context": [
#       "Maradona was an Argentinian Football Player"], "question": ["Who was Maradona?"]}))
