# MockFish
## Behavioural cloning of chess logic using convolutional neural networks.
### *Queen Mary University London MSc Data Analytics Final Project.*

## Abstract

Chess is an extremely complex game that requires deep forward thinking
and evaluation of future positions in order to master. However, when time
restrictions come into play, the strongest players must utilise rapid pattern
recognition and a ”gut feeling” that comes with years of practice.
With an approach building on the work of Oshri et.al. [8], I present
Mockfish, a pattern recognition chess engine built via behavioural cloning of
human players using only the board state as input.
By using a dataset of 45,000 games comprised of 1,700,000 moves, I am
able to train a series of convolutional neural networks to analyse board po-
sitions and present a probability distribution of favourable moves, using no
search or evaluation functions.
Despite not using these traditional chess engine bases, Mockfish has demon-
strated ability to beat Stockfish 15 up to skill level 11/20 at a search depth
of 6, when under equal time constraints.

## Playing Mockfish

The engine is playable through play_engine.ipynb if the model paths are fed into the required cells.
