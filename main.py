"""TODO: custom exceptions hierarchy
TODO: settings file/dict"""

import numpy as np
import random
import model_training
import implementation
import preparation
import data_loading
import stats
import data_filter
import pickle

seed = 3972  # int(time.time()*1000)%10000
print(f'Seed: {seed}')
np.random.seed(seed)
random.seed(seed)

LOAD_DATA = True
TRAIN_NEW_MODEL = False
CREATE_PLOTS = False
CREATE_NEW_API = False
USE_UI = True

EPOCHS = 10
LR_E = 0.004
LR_B = 0.005
LD_EU = 0.06
LD_EF = 0.06
DIM = 30

MIN_U = 20
MIN_F = 20

DB_PATH = "data/baza.db"

if LOAD_DATA or TRAIN_NEW_MODEL or CREATE_NEW_API or CREATE_PLOTS:
    data = data_loading.get_data(DB_PATH)
    if TRAIN_NEW_MODEL:
        data = data_filter.kcore_filter(data=data, min_u=MIN_U, min_f=MIN_F)
        tokenized_data, u2i, i2u, m2i, i2m, users, movies = preparation.tokenizer(data)

        train, val, test = preparation.train_val_test_split(tokenized_data, 0.8, 0.1, 0.1)

        mf_model = model_training.MF(train, val, test, dim=30, lr_embeddings=LR_E, lr_biases=LR_B,
                                     reg_film_embeddings=LD_EF,
                                     reg_user_embeddings=LD_EU)

        history1 = mf_model.train(epochs=EPOCHS)
        mf_model.save(path='models/model1.mf', u2i=u2i, i2u=i2u, m2i=m2i, i2m=i2m)
        with open('models/history.pkl', 'wb') as f:
            pickle.dump(history1, f)
        if CREATE_NEW_API:
            mf_model, u2i, i2u, m2i, i2m = model_training.MF.load('models/model1.mf')
            model_api = implementation.API(model=mf_model, u2i=u2i, m2i=m2i, data=tokenized_data)
            model_api.save('models/api.api')
    if CREATE_PLOTS:
        data_stats = stats.Stats(data)
        stats.print_df(data_stats.correlation_matrix('u'))
        with open('models/history.pkl', 'rb') as f:
            history1 = pickle.load(f)
        stats.plots(history1)

if USE_UI:
    ui = implementation.UI(api_path='models/api.api')
    ui.main()