> ### tf.keras.losses.CategoricalCrossentropy():
>
> ### one-hot多分类交叉熵损失

[官方tf.keras.losses.CategoricalCrossentropy](https://tensorflow.google.cn/api_docs/python/tf/keras/losses/CategoricalCrossentropy?hl=en)



**说明:**

Use this crossentropy loss function when there are two or more label classes. <u>We expect labels to be provided in a `one_hot` representation.</u> If you want to provide labels as integers, please use `SparseCategoricalCrossentropy` loss. There should be `# classes` floating point values per feature.

In the snippet below, there is `# classes` floating pointing values per example. The shape of both `y_pred` and `y_true` are `[batch_size, num_classes]`.

**格式:**

```python
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
    name='categorical_crossentropy'
)
```

**参数:**

- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a probability distribution. Note: Using from_logits=True may be more numerically stable.
- **label_smoothing**: Float in [0, 1]. When > 0, label values are smoothed, meaning the confidence on label values are relaxed. e.g. `label_smoothing=0.2` means that we will use a value of `0.1` for label `0` and `0.9` for label `1`"
- **reduction**: (Optional) Type of [`tf.keras.losses.Reduction`](https://tensorflow.google.cn/api_docs/python/tf/keras/losses/Reduction) to apply to loss. Default value is `AUTO`. `AUTO` indicates that the reduction option will be determined by the usage context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When used with [`tf.distribute.Strategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy), outside of built-in training loops such as [`tf.keras`](https://tensorflow.google.cn/api_docs/python/tf/keras) `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see https://www.tensorflow.org/tutorials/distribute/custom_training for more details on this.
- **name**: Optional name for the op.

**案例:**

```python
import tensorflow as tf

cce = tf.keras.losses.SparseCategoricalCrossentropy() # 类别是整型
loss = cce(
  tf.convert_to_tensor([0, 1, 2]),
  tf.convert_to_tensor([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]))
print('Loss: ', loss.numpy()) 
# Loss:  0.09458992
```



```python
cce = tf.keras.losses.CategoricalCrossentropy() # 类别需是one-hot类型
loss = cce(
  tf.convert_to_tensor(tf.one_hot([0, 1, 2], depth=len([0, 1, 2])) ),
  tf.convert_to_tensor([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]))
print('Loss: ', loss.numpy()) 
# Loss:  0.09458993
```



```python
# 自定义交叉熵损失
def cross_entropy(y_pred, y_true):
    # 把标签进行独热向量编码
    y_true = tf.one_hot(y_true, depth=num_classes)
    # 交叉熵损失
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))/num_classes

y_pred = [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]
y_true = [0,1,2]
num_classes = 3
print('Loss: ', cross_entropy(y_pred, y_true).numpy())
# Loss:  0.09458993
```

