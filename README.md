# PyTorch Classification Excercises (using Attention)

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
python main.py
```
