# PyTorch Classification Excercises

PyTorch demos and exercises written in pure Python. In these exrcises
we learn different text classifiers using and not using attention, and that
rely on various text encoding methods.

All models optimize a cross entropy mmuti-class/multinomial objective
(with each class encoded using one-hot vectors). We refer to the following definition
of cross entropy

$$
CE = - \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{C} ( y^{(i)}_j * \ln p^{(i)}_j )
$$

with $B$ being the batch size and $C$ the number of classes.

These experiments require a Python 3.10.x environment, and run on CPU. In order to run them, type:

```bash
pip install -r requirements.txt
python main.py
```

### Implementations

#### Attention Mechanisms

1) Classical self-attention (a.k.a. [hierarchical attention](https://aclanthology.org/N16-1174/)).
   We use it to implement attention on top of traditional RNN-based models over
   documents $d$ represented as sequences $w_1,\dots,w_t$ of words.
   In this version $h_i$ refers to
   the state of a hidden recursive unit at word/time $i$, for $0 \leq i \leq t$.
   The attention context $c$ becomes a $t$-dimensional vector that measures the contribution of
   each word $i$ in $d$ to the model's decision. In this setting, usually $c$ is fed into
   a softmax classification layer.

  $$
  \begin{align}
  u_i =& W \cdot h_i^T + b\\
  \alpha_i =& \text{softmax}( u_i \cdot u_i^T)\\
  c =& \sum_i \alpha_i \cdot u_i  
  \end{align}
  $$

2) Self-attention layer as in [transformer attention](https://arxiv.org/pdf/1706.03762). In this case, the context $c$ is a matrix or tensor,
and $h$ is a fully connected (hidden) layer. Additionally, $\text{softmax}( u \cdot u^T)$ can be normalized by the square root $\sqrt(d)$ of
trainable parameter $d$ (which we avoid for simplicity in this demo).
  
  $$
  \begin{align}
  u =& W \cdot h^T + b\\
  c =& \text{softmax}( u \cdot u^T) \cdot u
  \end{align}
  $$

#### Encodings

We implement two basic kinds of document encodings:
- Bag-of-words (BOW) embedding of documents.
- One-hot embeddings of documents.

#### Baselines

We implement two baselines for comparison:
- MLP for BOW embeddings.
- Convolutional neural network for the one-hot embeddings.
