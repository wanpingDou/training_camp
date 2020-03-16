> ### tf.data.Dataset.repeat()：
>
> ### 重复数据集元素。

[官方tf.data.Dataset.repeat](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en#repeat)



**说明：**

Repeats this dataset so each original value is seen `count` times.

**格式：**

```python
tf.data.Dataset.repeat(
    count=None
)
```
**参数：**

- **count**: (Optional.) A [`tf.int64`](https://tensorflow.google.cn/api_docs/python/tf#int64) scalar [`tf.Tensor`](https://tensorflow.google.cn/api_docs/python/tf/Tensor), representing the number of times the dataset should be repeated. The default behavior (if `count` is `None` or `-1`) is for the dataset be repeated indefinitely.

**返回:**

- **Dataset**: A `Dataset`.

**案例：**

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3]) 
dataset = dataset.repeat(2) 
for j,k in enumerate(dataset):
    print(j ,':' ,k.numpy())
```

> ```
> 0 : 1
> 1 : 2
> 2 : 3
> 3 : 1
> 4 : 2
> 5 : 3
> ```



