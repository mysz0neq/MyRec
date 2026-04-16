import numpy as np
import csv
from pathlib import Path
import random
import json
import time
import tqdm
import sqlite3

import tokenizer

seed = int(time.time()*1000)%10000
print(f'Seed: {seed}')
np.random.seed(seed)
random.seed(seed)

EPOCHS = 600
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
import tokenizer

data=data.get_data(DB_PATH)
#print(len(data))
data=filtr.filtr(data=data,min_u=MIN_U,min_f=MIN_F)
#print(len(data))
users,movies=tokenizer.users_movies_sets(data)
data,u2i,i2u,m2i,i2m=tokenizer.tokenizer(data)

import stats

#c_u,c_f=stats.counters(data)
##print(len(c_u),len(c_f))
#print(len(data))
#print(c_u)
data_stats= stats.Stats(data)

stats.print_df(data_stats.correlation_matrix('u'))


import mf_numpy

model=mf_numpy.MF(data,DIM)

model.optimizer(LR_E,LR_B,LD_EU,LD_EF)
model.train(EPOCHS)

