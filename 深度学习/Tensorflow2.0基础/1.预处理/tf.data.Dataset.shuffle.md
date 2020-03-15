> ### tf.data.Dataset.shuffle()：
>
> ### 随机打乱数据集元素。

[官方tf.data.Dataset.shuffle](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en#shuffle)



**说明：**

Randomly shuffles the elements of this dataset.

This dataset fills a buffer with `buffer_size` elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.

For instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1,000, then `shuffle` will initially select a random element from only the first 1,000 elements in the buffer. Once an element is selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element, maintaining the 1,000 element buffer.

**格式：**

```python
tf.data.Dataset.shuffle(buffer_size, 
                        seed=None, 
                        reshuffle_each_iteration=None
                       )
```
**参数：**

- **buffer_size**: A [`tf.int64`](https://tensorflow.google.cn/api_docs/python/tf#int64) scalar [`tf.Tensor`](https://tensorflow.google.cn/api_docs/python/tf/Tensor), representing the number of elements from this dataset from which the new dataset will sample.
- **seed**: (Optional.) A [`tf.int64`](https://tensorflow.google.cn/api_docs/python/tf#int64) scalar [`tf.Tensor`](https://tensorflow.google.cn/api_docs/python/tf/Tensor), representing the random seed that will be used to create the distribution. See [`tf.compat.v1.set_random_seed`](https://tensorflow.google.cn/api_docs/python/tf/compat/v1/set_random_seed) for behavior.
- **reshuffle_each_iteration**: (Optional.) A boolean, which if true indicates that the dataset should be pseudorandomly reshuffled each time it is iterated over. (Defaults to `True`.)

**返回:**

- **Dataset**: A `Dataset`.

**案例：**

```python
import tensorflow as tf

dt = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7])
dt0 = dt.shuffle(2)
for j,k in enumerate(dt0):
    print(j ,':' ,k.numpy())
```

> ```
> 0 : 2
> 1 : 3
> 2 : 4
> 3 : 5
> 4 : 1
> 5 : 6
> 6 : 7
> ```



