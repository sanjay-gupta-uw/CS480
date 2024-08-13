# Please note the following:
- I ran this on Colab initially (ran out of compute time) and converted it to run on windows (cuda)
     - Please note the following dependencies: python3.10 (compatible with tensorflow version used in model.ipynb)

- I implemented two models, the main one I worked on and is referenced in my report is defined in model.ipynb (this one had low score on leaderboard)
- main.ipynb had my high score, but I tried a different approach and ran out of time (hence the creation and performance of model.ipynb) (if u recreate this highscore, requires python 3.11)

- You may need to create the data/images directory containing test/train images (in proj folder)
- You may also need to define boosted_models directory in proj folder (code saves models to this dir during incremental x-boost training)
