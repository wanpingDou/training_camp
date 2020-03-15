# repeat-batch-shuffle案例

### 结论

> | 操作            | 结果                                       |
> | ------------- | ---------------------------------------- |
> | 先Shuffle      | 每个Repeat的顺序都不一样                          |
> | 后Shuffle      | 所有的Repeat顺序都不会变                          |
> | 先Repeat后Batch | 所有的Repeat数据放在一起再Batch，Batch_nums = ceil(data_nums x repeat_nums_/Batch_size) |
> | 先Batch后Repeat | 先把数据按照Batch_size分完，该步骤重复Repeat_nums次     |





##### 导入

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



##### 计数器

```python
lines = titanic_lines
counter = tf.data.experimental.Counter()
```



##### 1. Shuffle——》Repeat——》Batch

```python
lines4 = lines.take(4)
dataset = tf.data.Dataset.zip((counter, lines4))
shuffled = dataset.shuffle(buffer_size=3).repeat(2).batch(3)

n = 0
for i,j in shuffled:
    print(n,':  ', i.numpy())
    n+=1
```
> ```
> 0 :   [1 3 2]
> 1 :   [0 2 0]
> 2 :   [1 3]
> ```



##### 2. Shuffle——》Batch——》Repeat

```python
lines4 = lines.take(4)
dataset = tf.data.Dataset.zip((counter, lines4))
shuffled = dataset.shuffle(buffer_size=3).batch(3).repeat(2)

n = 0
for i,j in shuffled:
    print(n,':  ', i.numpy())
    n+=1
```

> ```
> 0 :   [2 0 1]
> 1 :   [3]
> 2 :   [0 2 1]
> 3 :   [3]
> ```



##### 3. Batch——》Repeat——》Shuffle

```python
lines4 = lines.take(4)
dataset = tf.data.Dataset.zip((counter, lines4))
shuffled = dataset.batch(3).repeat(2).shuffle(buffer_size=3)

n = 0
for i,j in shuffled:
    print(n,':  ', i.numpy())
    n+=1
```

> ```
> 0 :   [0 1 2]
> 1 :   [3]
> 2 :   [0 1 2]
> 3 :   [3]
> ```



##### 4. Repeat——》Batch——》Shuffle

```python
lines4 = lines.take(4)
dataset = tf.data.Dataset.zip((counter, lines4))
shuffled = dataset.repeat(2).batch(3).shuffle(buffer_size=3)

n = 0
for i,j in shuffled:
    print(n,':  ', i.numpy())
    n+=1
```

> ```
> 0 :   [0 1 2]
> 1 :   [3 0 1]
> 2 :   [2 3]
> ```



