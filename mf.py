"""(uid,fid,rtg)->wagi"""
from typing import Any

import numpy as np
from random import shuffle
from collections import defaultdict,Counter
import tokenizer
import matplotlib.pyplot as plt
import filtr

class MF:
    def __init__(self,
                 train_set:list,
                 test_set:list,
                 dim: int = 30,
                 lr_embeddings: float=0.004,
                 lr_biases:float=0.005,
                 reg_user_embeddings: float=0.06,
                 reg_film_embeddings: float=0.06):
        self.train_set = train_set
        self.test_set = test_set
        self.dim = dim
        self.lr_e=lr_embeddings
        self.lr_b=lr_biases
        self.ld_eu=reg_user_embeddings
        self.ld_ef=reg_film_embeddings

        self.users,self.films=tokenizer.users_movies_sets(train_set)
        self.counter_users,self.counter_films=filtr.counters(train_set)

        self.users_matrix,_=np.linalg.qr(np.random.normal(loc=0,
                                  scale=1,
                                  size=(len(self.users),self.dim)))
        self.films_matrix,_=np.linalg.qr(np.random.normal(loc=0,
                                    scale=1,
                                    size=(len(self.films), self.dim)))
        self.users_matrix*=0.001
        self.films_matrix*=0.001

        self.users_biases=np.zeros(shape=(len(self.users),))
        self.films_biases=np.zeros(shape=(len(self.films),))

        self.mean=np.mean([r for _,_,r in self.train_set])
    def predict(self,
                uid:int,
                fid:int) -> float:
        return np.dot(self.users_matrix[uid], self.films_matrix[fid]) + self.mean + self.users_biases[uid] + self.films_biases[fid]

    def train(self,
              epochs: int = 600):
        plt.figure(figsize=(10,7))
        x_labels=[]
        y_labels=[]
        prev_loss=float('inf')
        patience=0
        for epoch in range(epochs):
            shuffle(self.train_set)
            for uid,fid,rtg in self.train_set:
                der_bf, der_bu, grad_f, grad_u = self.step(fid, uid,rtg)

                self.users_matrix[uid]-=self.lr_e*grad_u/np.sqrt(self.counter_users[uid])
                self.films_matrix[fid]-=self.lr_e*grad_f/np.sqrt(self.counter_films[fid])
                self.users_biases[uid]-=self.lr_b*der_bu/np.sqrt(self.counter_users[uid])
                self.films_biases[fid]-=self.lr_b*der_bf/np.sqrt(self.counter_films[fid])
            loss = self.test_loss()
            if loss>prev_loss:
                patience+=1
                if patience>=5:
                    self.lr_b/=2
                    self.lr_e/=2
            else:
                patience=0
            prev_loss=loss
            if (epoch+1)%50==0:
                print(f"Epoch: {epoch+1}, loss: {loss}")
            x_labels.append(epoch)
            y_labels.append(loss)
        plt.plot(x_labels,y_labels)
        plt.show()

    def step(self, fid:int, uid:int, rtg:float) -> tuple[float, float, np.ndarray, np.ndarray]:
        user_emb = self.users_matrix[uid]
        film_emb = self.films_matrix[fid]

        pred = self.predict(uid, fid)
        real = rtg

        error = pred - real

        grad_u = 2 * error * film_emb + self.ld_eu * user_emb
        grad_f = 2 * error * user_emb + self.ld_ef * film_emb
        der_bu = 2 * error
        der_bf = 2 * error
        return der_bf, der_bu, grad_f, grad_u

    def test_loss(self) -> float:
        loss = 0
        for uid, fid, rtg in self.test_set:
            loss += (self.predict(uid,fid)-rtg) ** 2
        loss /= len(self.test_set)
        return loss

class MFExperimental(MF):
    def __init__(self,
                 train_set: list,
                 test_set: list,
                 dim: int = 30,
                 lr_embeddings: float = 0.004,
                 lr_biases: float = 0.005,
                 reg_user_embeddings: float = 0.06,
                 reg_film_embeddings: float = 0.06):
        super().__init__(train_set,test_set,dim,lr_embeddings,lr_biases,reg_user_embeddings,reg_film_embeddings)

        self.user_ratings = defaultdict(list)
        self.film_ratings = defaultdict(list)
        self.user_counters = dict()
        self.user_counter_normalized: defaultdict[int, dict] = defaultdict(dict)
        self.film_counters = dict()
        self.film_counter_normalized: defaultdict[int, dict] = defaultdict(dict)
        for u, f, o in self.train_set:
            self.user_ratings[u].append(o)
            self.film_ratings[f].append(o)
        for u, f in self.user_ratings.items():
            self.user_counters[u] = sorted([(r, c) for r, c in Counter(f).items()], key=lambda x: x[0])
            suma = sum(f)
            it = iter([(r, j / suma) for r, j in self.user_counters[u]])
            r, j = next(it)
            for i in range(21):
                self.user_counter_normalized[u][i / 2] = 0
                if r > i / 2:
                    continue
                else:
                    while r <= i / 2:
                        self.user_counter_normalized[u][i / 2] += j
                        try:
                            r, j = next(it)
                        except StopIteration:
                            r = float('inf')
                            break

        for u, f in self.film_ratings.items():
            self.film_counters[u] = sorted([(r, c) for r, c in Counter(f).items()], key=lambda x: x[0])
            suma = sum(f)
            it = iter([(r, j / suma) for r, j in self.film_counters[u]])
            r, j = next(it)
            for i in range(21):
                self.film_counter_normalized[u][i / 2] = 0
                if r > i / 2:
                    continue
                else:
                    while r <= i / 2:
                        self.film_counter_normalized[u][i / 2] += j
                        try:
                            r, j = next(it)
                        except StopIteration:
                            r = float('inf')
                            break

    def train(self,
              epochs: int = 600):
        plt.figure(figsize=(10, 7))
        x_labels = []
        y_labels = []
        prev_loss = float('inf')
        patience = 0
        for epoch in range(epochs):
            shuffle(self.train_set)
            for uid, fid, rtg in self.train_set:
                der_bf, der_bu, grad_f, grad_u,pred = self.step(fid, rtg, uid)

                rounded_pred = int(2 * pred) / 2
                rounded_pred = 10 if rounded_pred > 10 else 1 if rounded_pred < 1 else rounded_pred
                rounded_real = int(2 * rtg) / 2

                user_error_weight=abs(self.user_counter_normalized[uid][rounded_pred]-self.user_counter_normalized[uid][rounded_real])
                film_error_weight = abs(self.film_counter_normalized[fid][rounded_pred] - self.film_counter_normalized[fid][rounded_real])

                self.users_matrix[uid]-= self.lr_e * grad_u / np.sqrt(self.counter_users[uid])*user_error_weight
                self.films_matrix[fid]-= self.lr_e * grad_f / np.sqrt(self.counter_films[fid])*film_error_weight
                self.users_biases[uid] -= self.lr_b * der_bu / np.sqrt(self.counter_users[uid])*user_error_weight
                self.films_biases[fid] -= self.lr_b * der_bf / np.sqrt(self.counter_films[fid])*film_error_weight
            loss = self.test_loss()
            if loss > prev_loss:
                patience += 1
                if patience >= 3:
                    self.lr_b /= 2
                    self.lr_e /= 2
            else:
                patience = 0
            prev_loss = loss
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch + 1}, loss: {loss}")
            x_labels.append(epoch)
            y_labels.append(loss)
        plt.plot(x_labels, y_labels)
        plt.show()
    def step(self, fid:int, uid:int,rtg:float) -> tuple[float, float, np.ndarray, np.ndarray, float]:
        user_emb = self.users_matrix[uid]
        film_emb = self.films_matrix[fid]

        pred = self.predict(uid, fid)
        real = rtg

        error = pred - real

        grad_u = 2 * error * film_emb + self.ld_eu * user_emb
        grad_f = 2 * error * user_emb + self.ld_ef * film_emb
        der_bu = 2 * error
        der_bf = 2 * error
        return der_bf, der_bu, grad_f, grad_u, pred
