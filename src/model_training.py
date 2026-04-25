"""TODO: torch implementation
TODO: static counter - model_id"""
import warnings
from os import PathLike

import numpy as np
from random import shuffle
from copy import deepcopy
import pickle
from typing import Any
import preparation
import stats
import data_filter
import time


def create_matrix(nobj: list | set,
                  ndim: int,
                  scale: float) -> np.ndarray:
    """Initializes weight matrix using QR algorithm on columns in order to start training with embeddings that are fully PCA-proof.
    The training needs longer warm-up, but it assures that any correlation of components in trained matrix
    is the result of training and not initialization.\n
    \n
    Params:

    - 'nobj' list or set of entities that the matrix is being created for
    - 'ndim' dimension of embeddings
    - 'scale' non-zero value for initialization scale. The bigger it is, the longer the warm-up but also the higher chance of a total mess.

    Note: if 'ndim'>'nobj', the QR algorithm is not performed and returned matrix is initialized with just np.random.normal,
    however it is not advised to train a model that has higher embedding dimension than number of entities in the first place."""
    if len(nobj) > ndim:
        return np.asarray(scale * (np.linalg.qr(np.random.normal(loc=0,
                                                                 scale=1,
                                                                 size=(len(nobj), ndim)))[0]))
    elif len(nobj) == 0:
        return np.empty(shape=(0, ndim))
    else:
        return np.asarray(np.random.normal(loc=0, scale=scale, size=(len(nobj), ndim)))


class MF:
    """A classic matrix factorization model.
    It uses only NumPy implementation of SGD with a minor modification
    to minimize learning step for entities with higher number of records than the others."""
    counter = 0

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 test_set: list,
                 dim: int = 30,
                 lr_embeddings: float = 0.004,
                 lr_biases: float = 0.005,
                 reg_user_embeddings: float = 0.06,
                 reg_film_embeddings: float = 0.06,
                 scale: float = 0.001):
        MF.counter += 1
        self.model_id = MF.counter
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.dim = dim
        self.lr_e = lr_embeddings
        self.lr_b = lr_biases
        self.ld_eu = reg_user_embeddings
        self.ld_ef = reg_film_embeddings
        self.scale = scale

        self.users, self.films = preparation.users_movies_sets(train_set)
        self.counter_users, self.counter_films = data_filter.counters(train_set)

        self.users_matrix = create_matrix(self.users, self.dim, scale=self.scale)
        self.films_matrix = create_matrix(self.films, self.dim, scale=self.scale)

        self.users_biases = np.zeros(shape=(len(self.users),))
        self.films_biases = np.zeros(shape=(len(self.films),))

        self.mean = np.mean([r for _, _, r in self.train_set])

        self.baseline_loss = stats.baseline_loss(self.train_set, self.val_set)

    def predict(self,
                uid: int,
                fid: int) -> float:
        """Returns a value that the model thinks is correct for the provided input."""
        user_known = uid < self.users_matrix.shape[0]
        film_known = fid < self.films_matrix.shape[0]
        if user_known and film_known:
            return float(np.dot(self.users_matrix[uid], self.films_matrix[fid]) + self.mean + self.users_biases[uid] +
                         self.films_biases[fid])
        elif user_known and not film_known:
            warnings.warn("Movie not found. Returning mean average with user bias added.")
            return float(self.mean + self.users_biases[uid])
        elif not user_known and film_known:
            warnings.warn("User not found. Returning mean average with film bias added.")
            return float(self.mean + self.films_biases[fid])
        else:
            warnings.warn("User and movie not found. Returning mean average.")
            return float(self.mean)

    def update_weights(self,
                       uid: int,
                       fid: int,
                       grad_u: np.ndarray,
                       grad_f: np.ndarray,
                       der_bu: float,
                       der_bf: float,
                       user_modifier: float,
                       film_modifier: float,
                       lr_e: float,
                       lr_b: float) -> None:
        """Updates weights matrices.
        User and film modifiers are float values that change learning step for this specific update.
        They can be specified in numerous ways in calling methods."""
        self.users_matrix[uid] -= lr_e * grad_u * user_modifier
        self.films_matrix[fid] -= lr_e * grad_f * film_modifier
        self.users_biases[uid] -= lr_b * der_bu * user_modifier
        self.films_biases[fid] -= lr_b * der_bf * film_modifier
        return None

    def train(self,
              train_set: list[tuple[int, int, float]] = None,
              val_set: list[tuple[int, int, float]] = None,
              test_set: list[tuple[int, int, float]] = None,
              epochs: int = 600,
              patience_limit: int = 2) -> dict[str, Any]:
        """Performs training of the model. Returns dictionary with full history of the training:

        * 'overall_train_set_size', 'overall_val_set_size', 'overall_test_set_size' - sizes of sets specified in the __init__ method
        * 'current_train_set_size', 'current_val_set_size', 'current_test_set_size' - sized of sets specified in the params of this method
        * 'dim' - embedding dimension
        * 'number_of_users', 'number_of_films' - number of rows in weights matrices
        * 'lr_e', 'lr_b' - initial learning rates of embeddings and biases. Note that lr_b should be slightly higher than lr_e in order to get satisfying results
        * 'ld_eu', 'ld_ef' - lambda/regularization rates for user and film embeddings. The higher they are, the less expressive the embeddings are
        * 'x_labels' - list containing number of every epoch that has been performed - for plotting
        * 'train_labels', 'val_labels' - lists of train loss and validation loss values in every epoch that has been performed - for plotting
        * 'users_vectors', 'films_vectors' - lists of average norms of all embeddings in every epoch that has been performed - for plotting
        * 'users_biases', 'films_biases', 'users_abs_biases', 'films_abs_biases' - lists of average bias values and average absolute values in every epoch that has been performed - for plotting
        * 'baseline_loss' - loss value without any machine learning involved - computed using only average ratings of each user
        * 'test_loss' - loss value on test set computed on trained weights matrices
        * 'best_loss_epoch', 'best_loss' - epoch in which the lowest validation loss had been achieved and the value of this validation loss
        * 'lr_lowering_points' - list of numbers of epochs in which the learning rates had been lowered due to reaching some patience level
        * 'patience' - list of patience value in every epoch that has been performed - for plotting
        * 'model_id' - 0 unless you create more than one model in one run - for distinguishing models during testing
        \n
        Note: Sets in parameters of the method are for future fine-tuning implementation and are not necessary for initial training.
        That's the only part of the method that is fine-tuning friendly, so do not try to perform training that is not initial.
        """
        if train_set is None and val_set is None and test_set is None:
            train_set = self.train_set
            val_set = self.val_set
            test_set = self.test_set
        elif train_set is not None and val_set is not None and test_set is not None:
            pass
        else:
            raise Exception("You have to specify all three sets")
        lr_e = self.lr_e
        lr_b = self.lr_b

        history: dict[str, Any] = {
            'overall_train_set_size': len(self.train_set),
            'overall_val_set_size': len(self.val_set),
            'overall_test_set_size': len(self.test_set),
            'current_train_set_size': len(train_set),
            'current_val_set_size': len(val_set),
            'current_test_set_size': len(test_set),
            'dim': self.dim,
            'number_of_users': np.shape(self.users_matrix)[0],
            'number_of_films': np.shape(self.films_matrix)[0],
            'lr_e': lr_e,
            'lr_b': lr_b,
            'ld_eu': self.ld_eu,
            'ld_ef': self.ld_ef,
            'x_labels': [],
            'val_labels': [],
            'train_labels': [],
            'users_biases': [],
            'users_abs_biases': [],
            'films_biases': [],
            'films_abs_biases': [],
            'users_vectors': [],
            'films_vectors': [],
            'baseline_loss': self.baseline_loss,
            'test_loss': None,
            'best_loss': None,
            'best_loss_epoch': None,
            'lr_lowering_points': [],
            'patience': [],
            'model_id': self.model_id
        }

        best_user_matrix = deepcopy(self.users_matrix)
        best_film_matrix = deepcopy(self.films_matrix)
        best_user_biases = deepcopy(self.users_biases)
        best_film_biases = deepcopy(self.films_biases)

        prev_loss = float('inf')
        best_loss = float('inf')
        patience = 0

        print("\nStarting Training:")
        print("-" * 80)
        start_time = time.time()
        for epoch in range(epochs):
            shuffle(train_set)

            for uid, fid, rtg in train_set:
                self.step(fid, uid, rtg, lr_e=lr_e, lr_b=lr_b)

            val_loss = self.loss_on_set(val_set)
            train_loss = self.loss_on_set(train_set)
            if val_loss < best_loss:
                best_loss = val_loss
                best_user_matrix = deepcopy(self.users_matrix)
                best_film_matrix = deepcopy(self.films_matrix)
                best_user_biases = deepcopy(self.users_biases)
                best_film_biases = deepcopy(self.films_biases)
                history['best_loss'] = best_loss
                history['best_loss_epoch'] = epoch

            if epoch % 50 == 0 and (epoch + 1) > 150:
                if val_loss > prev_loss:
                    patience += 1
                    if patience >= patience_limit:
                        lr_b /= 2
                        lr_e /= 2
                        history['lr_lowering_points'].append(epoch)
                else:
                    patience = 0
                prev_loss = val_loss

            if lr_e < 1e-6:
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch: {epoch + 1:4} | Val. loss: {val_loss:5.4f} (best: {best_loss:5.4f}) | Train loss: {train_loss:5.4f} | Patience: {patience}")

            history['x_labels'].append(epoch)
            history['val_labels'].append(val_loss)
            history['train_labels'].append(train_loss)
            history['users_biases'].append(np.mean(self.users_biases))
            history['films_biases'].append(np.mean(self.films_biases))
            history['users_abs_biases'].append(np.mean(np.abs(self.users_biases)))
            history['films_abs_biases'].append(np.mean(np.abs(self.films_biases)))
            history['users_vectors'].append(np.mean(np.linalg.norm(self.users_matrix, axis=1)))
            history['films_vectors'].append(np.mean(np.linalg.norm(self.films_matrix, axis=1)))
            history['patience'].append(patience)

        self.users_matrix = best_user_matrix
        self.films_matrix = best_film_matrix
        self.users_biases = best_user_biases
        self.films_biases = best_film_biases
        history['test_loss'] = self.loss_on_set(self.test_set)
        end_time = time.time()
        print("-" * 80)
        print("Finished training:")
        print(f'Test loss: {self.loss_on_set(self.test_set):.4f} | Time: {end_time - start_time:.2f}s\n')

        return history

    def step(self,
             fid: int,
             uid: int,
             rtg: float,
             lr_e: float,
             lr_b: float) -> None:
        """Calculates gradients and derivatives for embeddings and biases alongside user and film modifiers. Updates weights matrices."""
        try:
            user_emb = self.users_matrix[uid]
            film_emb = self.films_matrix[fid]
        except IndexError as ie:
            print(f"{ie}\nIndexes fid={fid}, uid={uid}")
            return None

        pred = self.predict(uid, fid)
        real = rtg

        error = pred - real

        grad_u = 2 * error * film_emb + self.ld_eu * user_emb
        grad_f = 2 * error * user_emb + self.ld_ef * film_emb
        der_bu = 2 * error
        der_bf = 2 * error

        user_modifier = 1 / np.sqrt(self.counter_users[uid])
        film_modifier = 1 / np.sqrt(self.counter_films[fid])

        self.update_weights(uid, fid, grad_u, grad_f, der_bu, der_bf, user_modifier, film_modifier, lr_b=lr_b,
                            lr_e=lr_e)

        return None

    def loss_on_set(self,
                    which_set: list) -> float:
        """Calculates MSE loss value of the model on given set. Returns 0 if set is empty."""
        loss = 0
        if len(which_set) > 0:
            for uid, fid, rtg in which_set:
                loss += (self.predict(uid, fid) - rtg) ** 2
            loss /= len(which_set)
        return loss

    def save(self,
             path: int | str | bytes | PathLike[str] | PathLike[bytes],
             u2i: dict,
             i2u: dict,
             m2i: dict,
             i2m: dict) -> None:
        """Save a model and vocabulary dictionaries to the given path. Order in the tuple is: (model,u2i,i2u,m2i,i2m)."""
        with open(path, 'wb') as f:
            pickle.dump((self, u2i, i2u, m2i, i2m), f)
        print(f"Successfully saved the model to path {path}")
        return None

    @classmethod
    def load(cls,
             path: int | str | bytes | PathLike[str] | PathLike[bytes]) -> tuple['MF',dict,dict,dict,dict]:
        """Load a model and vocabulary dictionaries from the given path. Order in the tuple is: (model,u2i,i2u,m2i,i2m)
        Note: this is a static method, so loading isn't performed on class object, instead this method returns a saved object!"""
        with open(path, 'rb') as f:
            model, u2i, i2u, m2i, i2m = pickle.load(f)
        return model, u2i, i2u, m2i, i2m
