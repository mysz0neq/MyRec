"""TODO: custom exceptions hierarchy
TODO: settings file/dict"""

import numpy as np
from pathlib import Path
import random
import pickle




seed = 3972#int(time.time()*1000)%10000
print(f'Seed: {seed}')
np.random.seed(seed)
random.seed(seed)

EPOCHS = 1000
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
data,u2i,i2u,m2i,i2m,users,movies=preparation.tokenizer(data)

import stats
import mf

#c_u,c_f=stats.counters(data)
##print(len(c_u),len(c_f))
#print(len(data))
#print(c_u)
data_stats= stats.Stats(data)

#stats.print_df(new_data_stats.correlation_matrix('u'))

#stats.print_df(data_stats.correlation_matrix('u'))


#train,val,test=preparation.train_val_test_split(data,0.8,0.1,0.1)


#mf_model=mf.MF(train,val,test,dim=30,lr_embeddings=LR_E,lr_biases=LR_B,reg_film_embeddings=LD_EF,reg_user_embeddings=LD_EU)

#history1=mf_model.train(epochs=EPOCHS)
#mf_model.save('model1.mf')
mf_model=mf.MF.load('model1.mf')
ft_data=[
    ('nikola s','co sie zdarzylo baby jane',10),
    ('nikola s','bulwar zachodzacego slonca',10),
    ('nikola s','mulholland drive',10),
    ('matka','zabawa w pochowanego 2',7),
    ('muffinka1999','zabicie swietego jelenia',5),
    ('magic lve','jerry maguire',7),
    ('kacper traczykowski','osiem milimetrow',5),
    ('dajmian','blue moon',5),
    ('yune yamamoto','pomoc domowa',6),
    ('kamil pietrowski','ewolucja planety malp',7)
]
new_data,new_u2i,new_i2u,new_m2i,new_i2m,new_users,new_movies=preparation.tokenizer(ft_data,base_users=users,base_movies=movies,base_m2i=m2i,base_u2i=u2i)
new_data_stats=stats.Stats(new_data)
new_train,new_val,new_test=preparation.train_val_test_split(new_data,0.8,0.1,0.1)
history2=mf_model.fine_tune(new_train,new_val,new_test,1000)
mf_model.save('model2.mf')
import pickle
#with open('history.pkl','wb') as f:
    #pickle.dump((history1,history2),f)
#with open('history.pkl','rb') as f:
    #history1,history2=pickle.load(f)
#stats.plots(history1)
stats.plots(history2)