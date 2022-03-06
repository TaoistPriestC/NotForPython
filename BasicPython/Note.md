### (一)Numpy维度问题

#### (1.1) 自建数组

```python
# GenerallyConstructing data require the following steps:
# List -> np.array -> Tensor

```



#### (1.2) 数组与矩阵维度问题

使用numpy创建数组`X`，形状 `(x,)` 与`(x,1)`是不同的。前者是数组，元素个数`x`，后者是矩阵，行数`x`，每个行只有一个元素。此时想把数组转为矩阵可以使用 `reshape(-1,1)` 生成新对象，再拿这个引用覆盖原来的变量`X`即可。数组与矩阵进行四则运算，会触发广播机制。

```python
# (1)创建一个数组 x.shape = (3,)
x = np.array([1,2,3]) 

# (2)数组转为矩阵 X.shape = (3,1)
X = X.reshape(-1, 1)

# (3)矩阵转为数组 x.shape = (3,)
X = X.reshape(-1)

# (4)触发广播机制, 数组x会与y每列的元素发生广播并覆盖原来列上的元素
x = np.array([1, 2, 3])
y = x.reshape(-1, 1)
x + y = array([
    [3, 4, 5],   
    [4, 5, 6],
    [5, 6, 7]
])
```

