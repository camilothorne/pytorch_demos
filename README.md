# PyTorch Classification Excercises

PyTorch demos and exercises written in pure Python. In these exercises
we learn different text classifiers with and without attention, and that
rely on various text encoding methods.

All models optimize a cross entropy multinomial objective
(with each class encoded using one-hot vectors). We refer to the following definition
of cross entropy

$$
CE = - \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{C} ( y^{(i)}_j * \ln p^{(i)}_j )
$$

with $B$ being the batch size and $C$ the number of classes.

These experiments require a Python 3.10.x environment. They were tested on CPU and MPS
(Apple M2) architectures. To install dependencies, type
```bash
pip install -r requirements.txt
```
and to run the experiments, type
```
python main.py -e <exp_name> -i <iter> [-s <flag>]
```
`<exp_name>` should be one of `bow_base`, `one_hot_base`, `bow_att` or `one_hot_att` (a string), corresponding
to the implementations described below. `<iter>` refers to the number of epochs (an integer). Lastly, 
if option `-s` is set to (string) `yes`, the models will print feature and/or attention scores (depending on the model trained).
For more information type `python main.py -h`. Please note that you might need to run a comparatively high number of
training epochs until training with attention converges.

The results of the experiments (learning curves, predictions and F1-scores) will be logged on the `plots_and_stats` folder.

### Implementations

We implement two kind of input encodings, a sequential and a non-sequential representation, for which
we define different attention flavors. We also include two baselines for comparison. 

#### Attention Mechanisms

1) Classical self-attention (a.k.a. [hierarchical attention](https://aclanthology.org/N16-1174.pdf)).
   We use it to implement attention on top of traditional RNN-based models over
   documents $d$ represented as sequences $w_1,\dots,w_t$ of words.
   In this version $h_i$ refers to
   the network's recurrent hidden state hidden at word/time $i$, for $0 \leq i \leq t$.
   The attention context $c$ becomes a $t$-dimensional vector that measures the contribution of
   each word $i$ in $d$ to the model's decision. In this setting, usually $c$ is fed into
   a softmax classification layer:

  $$
  \begin{align*}
  u_i =& W \cdot h_i^T + b\\
  \alpha_i =& \text{softmax}( u_i \cdot u_i^T)\\
  c =& \sum_i \alpha_i \cdot u_i  
  \end{align*}
  $$

2) Self-attention layer as in [transformer attention](https://arxiv.org/pdf/1706.03762). In this case, the context
   $c$ is a matrix or tensor,
   $h$ is a fully connected (hidden) layerm and $c$ is usually fed into a linear (projection) layer
   whose output is fed to further downstream layers or directly into a softmax layer for classification.
   Additionally, $\text{softmax}( u \cdot u^T)$ can be normalized by
   the square root $\sqrt{d}$ of a trainable scaling parameter $d$, but we avoid this for simplicity in this demo:
  
  $$
  \begin{align*}
  u =& W \cdot h^T + b\\
  c =& \text{softmax}( u \cdot u^T) \cdot u
  \end{align*}
  $$

#### Encodings

1) Bag-of-words (BOW) embedding of documents.
2) One-hot embeddings of documents.

#### Baselines

1) MLP for BOW embeddings.
2) Convolutional neural network for the one-hot embeddings.
