> ### tf.keras.losses.Huber():
>
> ### 用于稳健回归(robust regression)，M估计法(M-estimator)和可加模型(additive model)。Huber损失的变体也可以用于分类。

[官方tf.keras.losses.Huber](https://tensorflow.google.cn/api_docs/python/tf/keras/losses/Huber?hl=en)



**说明:**

$$
loss=
\begin{cases}
0.5 * （y-f(x)）^2,   & \text{if |y-f(x)| <= d}\\
0.5 * d^2 + d * (|y-f(x)| - d),   & \text{if |y-f(x)| > d}
\end{cases}
$$
where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

Huber损失结合了MSE和MAE的最佳特性。对于较小的误差，它是二次的，否则是线性的(对于其梯度也是如此)。

当真实值与预测值之间误差较大时，采用线性损失。当真实值与预测值之间误差较小时，采用平方损失。

好处：1）对噪声不敏感；2）损失函数可导。

**格式:**

```python
tf.keras.losses.Huber(
    delta=1.0, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'
)
```

**参数:**

- **delta**: A float, the point where the Huber loss function changes from a quadratic to linear.
- **reduction**: (Optional) Type of [`tf.keras.losses.Reduction`](https://tensorflow.google.cn/api_docs/python/tf/keras/losses/Reduction) to apply to loss. Default value is `AUTO`. `AUTO` indicates that the reduction option will be determined by the usage context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When used with [`tf.distribute.Strategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy), outside of built-in training loops such as [`tf.keras`](https://tensorflow.google.cn/api_docs/python/tf/keras) `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see https://www.tensorflow.org/tutorials/distribute/custom_training for more details on this.
- **name**: Optional name for the op.

**案例:**

```python
import tensorflow as tf

l = tf.keras.losses.Huber()
loss = l([0., 1., 1.], [1., 0., 1.])
print('Loss: ', loss.numpy())  
# Loss: 0.333
```

