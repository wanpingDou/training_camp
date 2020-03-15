# repeat-batch案例



```python
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__

```



##### 读取数据

```python
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)
```

> ```
> Downloading data from https://storage.googleapis.com/tf-datasets/titanic/train.csv
> 32768/30874 [===============================] - 0s 0us/step
> ```



##### Repeat任务

```python
rp =titanic_lines.take(3).repeat(2)

k = 0
for i in rp:
    print(k, ': ', i.numpy())
    if (k+1)%3 == 0:
        print('--------------')
    k+=1
```

> ```
> 0 :  b'survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone'
> 1 :  b'0,male,22.0,1,0,7.25,Third,unknown,Southampton,n'
> 2 :  b'1,female,38.0,1,0,71.2833,First,C,Cherbourg,n'
> --------------
> 3 :  b'survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone'
> 4 :  b'0,male,22.0,1,0,7.25,Third,unknown,Southampton,n'
> 5 :  b'1,female,38.0,1,0,71.2833,First,C,Cherbourg,n'
> --------------
> ```



##### Batch任务

```python
epochs = 3
dataset = titanic_lines.batch(300)

for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)
    print("End of epoch: ", epoch)
```
> ```
> (300,)
> (300,)
> (28,)
> End of epoch:  0
> (300,)
> (300,)
> (28,)
> End of epoch:  1
> (300,)
> (300,)
> (28,)
> End of epoch:  2
> ```



##### 先Repeat后Batch

```python
def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
```

```python
titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)
```

![repeat_batch](img/repeat_batch.png)

##### 先Batch后Repeat

```python
titanic_batches = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batches)
```

![batch_repeat](img/batch_repeat.png)