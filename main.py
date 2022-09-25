from fastapi import FastAPI, Query, HTTPException
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

from math import ceil


MODELS_DIR = 'checkpoints/'
app = FastAPI()

class TweetIn(BaseModel):
    tweets: List[str]

class TweetOut(BaseModel):
    forecast: dict

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


@app.post("/predict", response_model=TweetOut, status_code=200)
def get_prediction(tweets_in: TweetIn):

    device = "cpu"
    tweets = tweets_in.tweets

    ins = []
    for i, t in enumerate(tweets):
        ins.append(types.Tweet(i, t))
    ins = [types.Tweet(0, tweets)]

    predictions = {}
    for name in MODELS:
        model = MODELS[name]
        predictions[name] = model.predict(ins)

    [test_inputs] = get_scores(model_names=MODELS.keys(), datasets=[ins],
                    batch_size=batch_size, device=device, model_dir=MODELS_DIR, data_tag="english")
    aggregator = train_aggregator.MetaClassifier(num_models=8, hidden_size=hidden_size, dropout=dropout).to(device)
    aggregator.load_state_dict(torch.load(MODELS_DIR+'/aggregator-english/model.p', map_location=torch.device(device)))
    predictions['aggregator'] = aggregator.predict(test_inputs.to(device))

    if not predictions:
        raise HTTPException(status_code=400, detail="Model not found.")

    print(predictions)
    return predictions