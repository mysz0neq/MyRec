# Movie recommendation engine with data analysis tools

A fully custom-built recommendation system, created entirely from scratch using only NumPy and Pandas, without any high-level ML frameworks like PyTorch.

The first lines of code in my life were written for this project, so it has been evolving with me for the last year since February 2025. This repository is a cleaned-up version of the project.

## Key features and engineering decisions:
* **Custom machine learning engine:** Matrix Factorization implemented via Stochastic Gradient Descent (SGD) in pure NumPy. Features dynamic learning rate, early stopping, and L2 regularization.
* **PCA-proof Initialization:** Weight matrices are initialized using the QR algorithm to ensure orthogonal and independent components from epoch 0, preventing any correlation in the warm-up phase.
* **Group recommendations:** Implemented a group recommendation algorithm tailored for multiple users (e.g. couples or friends). It sorts recommendations by minimal predicted rating out of the entire group, which I found to be the best solution, and later realised that it's been actually scientifically proven to be the best solution (called "Least misery" strategy).
* **API and demo console UI:** In the `implementation.py` file there are API and UI classes ready for interaction with the user.
* **Vectorized statistics:** Stats calculations, rankings, and correlation matrices are handled by Pandas `DataFrame` operations for maximum performance. 
* **Training history:** `.train()` method returns a history dictionary which can be integrated with the `stats.plots()` method for creating plots and their finite-difference plots.

## Usage:

To run the project, first you need to specify the DB source. `data_loading.py` is created for fetching data from an SQLite database in which ratings data is in a table named "oceny" (Polish word for "ratings"), first column contains usernames/IDs, second column contains movie titles/IDs, and third column contains ratings. If your data is in a different format, you have to override this method, but keep in mind that every algorithm in the project follows the (u, m, r) format.
When you have the data ready, all you need to do is simply configure the flags at the top of `main.py` and execute the script.

## Known limitations and TODOs:
* **Console UI:** The current user interface is a CLI loop meant strictly for demonstration purposes. In the future I plan on building a real frontend for a web application like Letterboxd.
* **Fine-tuning:** In the code there are leftovers of an experimental fine_tune method (second-time tokenization, some exception handling that can only occur during fine_tuning, etc.). The method had been disabled for this branch for stability purposes, however I still plan to develop it in the future. The reason behind removing the method is that I realized that I need a much better DB schema than the one I have in my SQLite dataset - first and foremost there's a necessity for holding ratings dates, which I don't have in my custom scraped dataset.
* **Modifying loss function:** As with the fine-tuning method, I also plan to modify the loss function. I noticed that in the case of a non-uniform ratings distribution (which is 99.99% of the distributions in rating systems), the MSE Loss function is the wrong choice, so I plan on some kind of weighting the loss value based on the rating that has been predicted (i.e. predicting 7 instead of 8 is a much worse mistake than predicting 1 instead of 2 - because the most common ratings are 7s and 8s, whereas 1s and 2s are not so common).
* **Custom models testing framework:** I also plan on building another framework from scratch, which will allow hyperparameter tuning. With this framework I will be able to compare the performance of this model with the experimental one mentioned in the bullet point above, and also with a Torch implementation.
* **LLM integration in the distant future:** With a working and stable recommendation engine in the planned web app, I plan to implement another custom-made low-level ML project - this time a BERT-like LLM for user review analysis.
