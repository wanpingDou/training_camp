### 张量

> 多维数组或高维矩阵。

facebook的Pytorch，Keras仿照它开发的。

- 具有强大的GPU加速的张量计算。
- 包含自动求导系统的深度学习网络。

### 操作



```python
import torch

x = torch.empty(5,3) # 5x3的0矩阵（浮点）
x = torch.rand(5,3)  # 5x3的随机矩阵
x = torch.zeros(5,3) # 5x3的全0矩阵（整数）
```



###### 初始化Tensor

```python
# list创建
x = torch.tensor([3, 2, 4])
# new_ones创建
x = x.new_ones(5, 3, dtype=torch.double)
# randn_like创建
x =  torch.randn_like(x, dtype=torch.float)
```



###### 获取Size

```python
# 相当于numpy的shape
x.size()
```



###### 运算

> - 广播特性
> - 所有以_为结尾的操作都会以结果替换原变量

```python
# 加法1
x + y
# 加法2
torch.add(x,y)
# 提供输出tensor作为参数
reslut = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 替换
y.add_(x) # 将x加到y上去，相当于y=y+x
print(y)
```



###### 切片操作

```python
x[:, 1]
```



###### Reshape

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.videw(-1, 8)
print(x.size(), y.size())
```



###### 单元素取值

```python
x = torch.randn(2)
x.item()
# 0.23423
```



###### Tensor转Numpy

```python
a = torch.ones(5)
b = a.numpy() # b是数组
```



###### Numpy转Tensor

```python
import numpy as np
a = np.ones(5) # a是数组
b = torch.from_numpy(a) # 将数组a转换为tensor
```



###### CUDA张量

```python
# 使用.to方法将Tensor移动到任何设备中
# is_available 函数判断是否有cuda可用
# torch.device 将张量移动到指定的设备

if torch.cuda.is_avaiable():
    device = torch.device("cuda") # cuda设备对象
    y = torch.ones_like(x, device=device) # 直接从GPU创建张量
    x = x.to(device) # 将张量移动到cuda
    z = x + y 
    z.to("cpu", torch.double) # 也可对变量类型做更改
```



