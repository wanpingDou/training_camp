> ### tf.matmul()：
>
> ### 将矩阵a乘以矩阵b，生成a * b，a和b数据类型要一致。

[官方tf.linalg.matmul](https://tensorflow.google.cn/api_docs/python/tf/linalg/matmul?hl=en)可简写成tf.matmul



**格式:**

```python
tf.matmul(a, b, 
          transpose_a=False, 
          transpose_b=False, 
          adjoint_a=False, 
          adjoint_b=False, 
          a_is_sparse=False, 
          b_is_sparse=False, 
          name=None)
```

**参数:**

- **a**: 一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量。
- **b**: 一个类型跟张量a相同的张量。
- **transpose_a**: 如果为真，a则在进行乘法计算前进行转置。
- **transpose_b**: 如果为真， b则在进行乘法计算前进行转置。
- **adjoint_a**: 如果为真， a则在进行乘法计算前进行共轭和转置。
- **adjoint_b**: 如果为真，b则在进行乘法计算前进行共轭和转置。
- **a_is_sparse**: 如果为真，a会被处理为稀疏矩阵。
- **b_is_sparse**: 如果为真，b会被处理为稀疏矩阵。
- **name**: 操作的名字（可选参数）。



**返回值：**

- 一个跟张量a和张量b类型一样的张量。

**注意：**

（1）输入必须是矩阵（或者是张量秩 >２的张量，表示成批的矩阵），并且其在转置之后有相匹配的矩阵尺寸。
（2）两个矩阵必须都是同样的类型，支持的类型如下：float16, float32, float64, int32, complex64, complex128。

**案例：**

```python
import tensorflow as tf
import numpy as np

a = np.array(list(range(12)), np.float32)
a = a.reshape([3, -1])

b = np.array(list(range(12)), np.float32)
b = b.reshape([4, -1])

tf.matmul(a,b)
```

> ```latex
> <tf.Tensor: id=45, shape=(3, 3), dtype=float32, numpy=
> array([[ 42.,  48.,  54.],
>        [114., 136., 158.],
>        [186., 224., 262.]], dtype=float32)>
> ```