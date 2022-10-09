from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from typing import List, Tuple, Callable, TypeVar, Any, overload, Dict
from typing import Optional as Maybe

import torch
from torch import Tensor, LongTensor
from torch import load, cat, stack, save, no_grad, manual_seed
from torch.cuda import empty_cache

from nlp4ifchallenge import types
from nlp4ifchallenge.scripts import train_bert, train_aggregator
from nlp4ifchallenge.models import bert, aggregation

from transformers.utils import logging
logging.set_verbosity("CRITICAL")

from lime.lime_text import LimeTextExplainer

from math import ceil
import numpy as np


MODELS_DIR = 'checkpoints/'
app = FastAPI()

class TweetIn(BaseModel):
    tweets: List[str]

# class TweetOut(BaseModel):
#     forecast: dict

device = "cpu"
batch_size = 16
ignore_nan = False
hidden_size = 12
dropout = 0.25

MODELS = {'vinai-covid': None,
          'vinai-tweet': None,
          'cardiffnlp-tweet': None,
          'cardiffnlp-hate': None,
          'del-covid': None,
          'cardiffnlp-irony': None,
          'cardiffnlp-offensive': None,
          'cardiffnlp-emotion': None}

CHOSEN_MODEL = 'cardiffnlp-tweet'


for name in MODELS:
    model = bert.make_model(name=name, ignore_nan=ignore_nan)
    checkpoint = torch.load(MODELS_DIR+name+'-english/model.p', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    MODELS[name] = model


def get_scores(model_names: List[str], datasets: List[List[types.Tweet]], batch_size: int, device: str,
               model_dir: str, data_tag: str) -> List[Tensor]:
    """
       :returns num_dataset tensors of shape B x M x Q
    """
    outs = []
    for name in model_names:
        this_model_outs = []
        model = MODELS[name]
        for dataset in datasets:
            this_dataset_outs = []
            nbatches = ceil(len(dataset) / batch_size)
            for batch_idx in range(nbatches):
                start, end = batch_idx * batch_size, min(len(dataset), (batch_idx + 1) * batch_size)
                this_dataset_outs.append(model.predict_scores(dataset[start:end]).cpu())
            this_model_outs.append(cat(this_dataset_outs, dim=0))
        outs.append(this_model_outs)
        empty_cache()
    
    return [stack(x, dim=1) for x in zip(*outs)]


def get_scores_all_models(tweets):
    ins = []
    for i, t in enumerate(tweets):
        ins.append(types.Tweet(i, t))

    predictions = {}
    for name in MODELS:
        model = MODELS[name]
        predictions[name] = model.predict_scores(ins).cpu().data.numpy()

    [test_inputs] = get_scores(model_names=MODELS.keys(), datasets=[ins],
                    batch_size=batch_size, device=device, model_dir=MODELS_DIR, data_tag="english")
    aggregator = train_aggregator.MetaClassifier(num_models=8, hidden_size=hidden_size, dropout=dropout).to(device)
    aggregator.load_state_dict(torch.load(MODELS_DIR+'/aggregator-english/model.p', map_location=torch.device(device)))
    predictions['aggregator'] = aggregator.forward(test_inputs.to(device)).sigmoid().cpu().data.numpy()

    return predictions


def get_scores_for_explainer(tweets):
    pred = get_scores_all_models(tweets)[CHOSEN_MODEL]
    r = []
    for i in pred:
        temp = i[1]
        r.append(np.array([1-temp-0.4, temp+0.4])) 

    return np.array(r)


def explain_label(arr):
    explainer = LimeTextExplainer(class_names=['no', 'yes'], random_state=2)
    outs = []

    for tweet in arr:
        exp = explainer.explain_instance(
            tweet,
            get_scores_for_explainer,
            num_features=10,
            num_samples=10)
        outs.append(exp.as_html(text=tweet))

    return outs


@app.post("/predict", status_code=200)
def get_prediction(tweets_in: TweetIn):
    tweets = tweets_in.tweets

    html_response = explain_label(tweets)
    if not html_response:
        raise HTTPException(status_code=400, detail="Model not found.")

    return HTMLResponse(" ".join(html_response))
