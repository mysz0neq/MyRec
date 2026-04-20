"""TODO: fine_tune, torch implementation, clean-up of the code, saving and loading trained model"""
from argparse import ArgumentError

import numpy as np
from random import shuffle
from collections import defaultdict,Counter
from copy import deepcopy

import stats
import preparation
import filtr
import time


def create_matrix(nobj: list | set, ndim:int,scale) -> np.ndarray:
    return np.linalg.qr(np.random.normal(loc=0,
                                scale=1,
                                size=(len(nobj), ndim)))[0]*scale


class MF:
    def __init__(self,
                 train_set:list,
                 val_set:list,
                 test_set:list,
                 dim: int = 30,
                 lr_embeddings: float=0.004,
                 lr_biases:float=0.005,
                 reg_user_embeddings: float=0.06,
                 reg_film_embeddings: float=0.06,
                 scale:float=0.001):
        self.train_set = train_set
        self.val_set=val_set
        self.test_set = test_set
        self.dim = dim
        self.lr_e=lr_embeddings
        self.lr_b=lr_biases
        self.ld_eu=reg_user_embeddings
        self.ld_ef=reg_film_embeddings
        self.scale=scale

        self.users,self.films=preparation.users_movies_sets(train_set)
        self.counter_users,self.counter_films=filtr.counters(train_set)

        self.users_matrix=create_matrix(self.users,self.dim,scale=self.scale)
        self.films_matrix=create_matrix(self.films,self.dim,scale=self.scale)

        self.users_biases=np.zeros(shape=(len(self.users),))
        self.films_biases=np.zeros(shape=(len(self.films),))

        self.mean=np.mean([r for _,_,r in self.train_set])

        self.baseline_loss=stats.baseline_loss(self.train_set,self.val_set)

    def predict(self,
                uid:int,
                fid:int) -> float:
        return np.dot(self.users_matrix[uid], self.films_matrix[fid]) + self.mean + self.users_biases[uid] + self.films_biases[fid]
    def update_weights(self,
                       uid,
                       fid,
                       grad_u,
                       grad_f,
                       der_bu,
                       der_bf,
                       user_modifier,
                       film_modifier):
        self.users_matrix[uid] -= self.lr_e * grad_u *user_modifier
        self.films_matrix[fid] -= self.lr_e * grad_f *film_modifier
        self.users_biases[uid] -= self.lr_b * der_bu *user_modifier
        self.films_biases[fid] -= self.lr_b * der_bf *film_modifier

    def train(self,
              train_set:list[tuple[int,int,float]]=None,
              val_set:list[tuple[int,int,float]]=None,
              test_set:list[tuple[int,int,float]]=None,
              epochs: int = 600):
        if not train_set and not val_set and not test_set:
            train_set=self.train_set
            val_set=self.val_set
            test_set=self.test_set
        elif train_set and val_set and test_set:
            self.train_set+=train_set
            self.val_set+=val_set
            self.test_set+=test_set
        else:
            raise Exception("You have to specify all three sets")


        history={
            'x_labels':[],
            'val_labels':[],
            'train_labels':[],
            'users_biases':[],
            'users_abs_biases':[],
            'films_biases':[],
            'films_abs_biases':[],
            'users_vectors':[],
            'films_vectors':[],
            'baseline_loss':self.baseline_loss,
            'test_loss':None
        }

        best_user_matrix = deepcopy(self.users_matrix)
        best_film_matrix = deepcopy(self.films_matrix)
        best_user_biases = deepcopy(self.users_biases)
        best_film_biases = deepcopy(self.films_biases)

        prev_loss=float('inf')
        best_loss=float('inf')
        patience=0


        print("\nStarting Training:")
        print("-" * 80)
        start_time=time.time()
        for epoch in range(epochs):
            shuffle(train_set)

            for uid,fid,rtg in train_set:
                self.step(fid, uid,rtg)

            val_loss = self.loss_on_set(val_set)
            train_loss = self.loss_on_set(train_set)
            if val_loss < best_loss:
                best_loss = val_loss
                best_user_matrix = deepcopy(self.users_matrix)
                best_film_matrix = deepcopy(self.films_matrix)
                best_user_biases = deepcopy(self.users_biases)
                best_film_biases = deepcopy(self.films_biases)

            if epoch%50 == 0 and (epoch+1)>200:
                if val_loss>prev_loss:
                    patience+=1
                    if patience>=2:
                        self.lr_b/=2
                        self.lr_e/=2
                else:
                    patience=0
                prev_loss=val_loss

            if self.lr_e<1e-6:
                break

            if (epoch+1)%50==0:
                print(f"Epoch: {epoch+1:4} | Val. loss: {val_loss:5.4f} (best: {best_loss:5.4f}) | Train loss: {train_loss:5.4f} | Patience: {patience}")

            history['x_labels'].append(epoch)
            history['val_labels'].append(val_loss)
            history['train_labels'].append(train_loss)
            history['users_biases'].append(np.mean(self.users_biases))
            history['films_biases'].append(np.mean(self.films_biases))
            history['users_abs_biases'].append(np.mean(np.abs(self.users_biases)))
            history['films_abs_biases'].append(np.mean(np.abs(self.films_biases)))
            history['users_vectors'].append(np.mean(np.linalg.norm(self.users_matrix,axis=1)))
            history['films_vectors'].append(np.mean(np.linalg.norm(self.films_matrix,axis=1)))


        self.users_matrix=best_user_matrix
        self.films_matrix=best_film_matrix
        self.users_biases=best_user_biases
        self.films_biases=best_film_biases
        history['test_loss'] = self.loss_on_set(self.test_set)
        end_time=time.time()
        print("\n","-"*80)
        print("Finished training:")
        print(f'Test loss: {self.loss_on_set(self.test_set):.4f} | Time: {end_time-start_time:.2f}s')

        return history

    def step(self, fid:int, uid:int, rtg:float) -> None:
        user_emb = self.users_matrix[uid]
        film_emb = self.films_matrix[fid]

        pred = self.predict(uid, fid)
        real = rtg

        error = pred - real

        grad_u = 2 * error * film_emb + self.ld_eu * user_emb
        grad_f = 2 * error * user_emb + self.ld_ef * film_emb
        der_bu = 2 * error
        der_bf = 2 * error

        user_modifier = 1 / np.sqrt(self.counter_users[uid])
        film_modifier = 1 / np.sqrt(self.counter_films[fid])

        self.update_weights(uid, fid, grad_u, grad_f, der_bu, der_bf, user_modifier, film_modifier)

        return None
    def loss_on_set(self,which_set:list) -> float:
        loss = 0
        for uid, fid, rtg in which_set:
            loss += (self.predict(uid, fid) - rtg) ** 2
        loss /= len(which_set)
        return loss

    def fine_tune(self,
                  new_tokenized_train:list[tuple[int,int,float]],
                  epochs:int = 100):
        """Perform training on previously trained model using only new data:

            * ratings from newly added users and films:
                1. Extends weight matrices
                2. Trains only new embeddings, freezing old ones

            * new ratings from already existing users

        FIXME: train,val,test sets"""
        users,movies=preparation.users_movies_sets(new_tokenized_train)
        new_users=users-self.users
        new_movies=movies-self.films

        self.users_matrix=np.stack([self.users_matrix,create_matrix(new_users,self.dim,scale=self.scale)])
        self.films_matrix = np.stack([self.films_matrix, create_matrix(new_movies, self.dim, scale=self.scale)])

        data_from_new=[(u,f,o) for u,f,o in new_tokenized_train if u in new_users or f in new_movies]

        data_from_old=[(u,f,o) for u,f,o in new_tokenized_train if u not in new_users and f not in new_movies]

        history=self.train(epochs=600,
                           train_set=data_from_new)

        raise NotImplementedError
    def save(self,path):
        raise NotImplementedError
    def load(self,path):
        raise NotImplementedError

class MFExperimental(MF):
    """Weighing errors by how big a difference it really is for the user.
    Problem is I don't really know how to compare those two architectures so for now it remains experimental.

    Disclaimer: Needs a lot bigger learning rates than superclass, however I didn't figure out how much bigger."""
    def __init__(self,
                 train_set: list,
                 val_set:list,
                 test_set: list,
                 dim: int = 30,
                 lr_embeddings: float = 4,
                 lr_biases: float = 5,
                 reg_user_embeddings: float = 0.06,
                 reg_film_embeddings: float = 0.06,
                 scale:float=0.01) -> None:
        super().__init__(train_set,val_set,test_set,dim,lr_embeddings,lr_biases,reg_user_embeddings,reg_film_embeddings,scale)

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
    def step(self, fid:int, uid:int, rtg:float) -> None:
        user_emb = self.users_matrix[uid]
        film_emb = self.films_matrix[fid]

        pred = self.predict(uid, fid)
        real = rtg

        error = pred - real

        grad_u = 2 * error * film_emb + self.ld_eu * user_emb
        grad_f = 2 * error * user_emb + self.ld_ef * film_emb
        der_bu = 2 * error
        der_bf = 2 * error

        rounded_pred = int(2 * pred) / 2
        rounded_pred = 10 if rounded_pred > 10 else 1 if rounded_pred < 1 else rounded_pred
        rounded_real = int(2 * rtg) / 2

        user_modifier = abs(
            self.user_counter_normalized[uid][rounded_pred] - self.user_counter_normalized[uid][rounded_real]) / np.sqrt(self.counter_users[uid])
        film_modifier = abs(
            self.film_counter_normalized[fid][rounded_pred] - self.film_counter_normalized[fid][rounded_real]) / np.sqrt(self.counter_films[fid])

        self.update_weights(uid, fid, grad_u, grad_f, der_bu, der_bf, user_modifier, film_modifier)
        return None