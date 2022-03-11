<h2 ><center  style="color:red;">实干Python笔记</center></h2>



### (一)科学计算三剑客

#### (1.1)Numpy常见问题

**[01]** 使用Torch进行深度学习任务，要把数组转化张量！

```python
# Constructing data require the following steps:
# List -> np.array -> Tensor
```

**[02]** 形状 `(x,)` 与`(x,1)`是不同的。前者是数组，后者是矩阵！

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
""" 
	这种数组与矩阵的维度问题也会出现于torch框架中，比如下列这个经典的报错信息:
	UserWarning: Using a target size (torch.Size([16])) that is different to the input size (torch.Size([16, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
	解决办法是把前向传播的预测值，通过 sequeeze 函数压扁，如果这个过程已被封装，则在`forward` 函数返回运算结果之前把矩阵压成数组。
"""
```



