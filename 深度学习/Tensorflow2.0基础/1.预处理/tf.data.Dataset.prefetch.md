> ### tf.data.Dataset.prefetch()：
>
> ### 预取数据集元素。

[官方tf.data.Dataset.prefetch](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en#prefetch)



**说明：**

Creates a `Dataset` that prefetches elements from this dataset.

Most dataset input pipelines should end with a call to `prefetch`. <u>This allows later elements to be prepared while the current element is being processed.</u> This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

**格式：**

```python
prefetch(
    buffer_size
)
```

**参数：**

- **buffer_size**: A [`tf.int64`](https://tensorflow.google.cn/api_docs/python/tf#int64) scalar [`tf.Tensor`](https://tensorflow.google.cn/api_docs/python/tf/Tensor), representing the maximum number of elements that will be buffered when prefetching.

**返回:**

- **Dataset**: A `Dataset`.

**注意：**

- Like other `Dataset` methods, prefetch operates on the elements of the input dataset.   It has no concept of examples vs. batches.         
- `examples.prefetch(2)` will prefetch two elements (2 examples), while `examples.batch(20).prefetch(2)` will prefetch 2 elements (2 batches, of 20 examples each).

**案例：**

```python
import tensorflow as tf
import time

# 构建数据
dt = tf.data.Dataset.from_tensor_slices(list(range(1000000)))

# 带prefetch的批次
dt0 = dt.batch(10).prefetch(5000)
s = time.clock()
for j,k in enumerate(dt0):
    if j%100000==0:
        print(j ,':' ,k.numpy())
print("总耗时： ", time.clock()-s)
# 0 : [0 1 2 3 4 5 6 7 8 9]
# 总耗时：  5.449731500000041

# 不带prefetch的批次
dt0 = dt.batch(10)
s = time.clock()
for j,k in enumerate(dt0):
    if j%100000==0:
        print(j ,':' ,k.numpy())
print("总耗时： ", time.clock()-s)
# 0 : [0 1 2 3 4 5 6 7 8 9]
# 总耗时：  5.820961699999998
```



