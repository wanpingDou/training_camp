> ### tf.data.Dataset.skip(count)：
>
> ### 跳过数据集count个元素。

[官方tf.data.Dataset.count](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en#count)



**说明：**

Creates a `Dataset` that skips `count` elements from this dataset.

**格式：**

```python
tf.data.Dataset.skip(
    count
)
```
**参数：**

- **count**: A [`tf.int64`](https://tensorflow.google.cn/api_docs/python/tf#int64) scalar [`tf.Tensor`](https://tensorflow.google.cn/api_docs/python/tf/Tensor), representing the number of elements of this dataset that should be skipped to form the new dataset. If `count` is greater than the size of this dataset, the new dataset will contain no elements. If `count` is -1, skips the entire dataset.

**返回:**

- **Dataset**: A `Dataset`.

**案例：**

```python
import tensorflow as tf

dt = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7])
dt0 = dt.skip(3)
for j,k in enumerate(dt0):
    print(j ,':' ,k.numpy())
```

> ```
> 0 : 4
> 1 : 5
> 2 : 6
> 3 : 7
> ```



