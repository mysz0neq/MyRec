import numpy as np
import csv
from pathlib import Path
import random
import json
import time
import tqdm
import sqlite3

import preparation

seed = int(time.time()*1000)%10000
print(f'Seed: {seed}')
np.random.seed(seed)
random.seed(seed)

EPOCHS = 5
LR_E=0.004
LR_B=0.005
LD_EU=0.06
LD_EF=0.06
DIM=30

MIN_U=20
MIN_F=20

PATH=Path(__file__).parent
DB_PATH=PATH.joinpath("baza.db")

import data
import filtr
import preparation

data=data.get_data(DB_PATH)
#print(len(data))
data=filtr.filtr(data=data,min_u=MIN_U,min_f=MIN_F)
#print(len(data))
users,movies=preparation.users_movies_sets(data)
data,u2i,i2u,m2i,i2m=preparation.tokenizer(data)

import stats
import mf

#c_u,c_f=stats.counters(data)
##print(len(c_u),len(c_f))
#print(len(data))
#print(c_u)
data_stats= stats.Stats(data)

#stats.print_df(data_stats.correlation_matrix('u'))


train,val,test=preparation.train_val_test_split(data,0.8,0.1,0.1)

mf_model=mf.MF(train,val,test,dim=30,lr_embeddings=LR_E,lr_biases=LR_B,reg_film_embeddings=LD_EF,reg_user_embeddings=LD_EU)

history=mf_model.train(EPOCHS)
print(mf_model.fine_tune([(1345,45391,2)],5))
stats.plots(history)

