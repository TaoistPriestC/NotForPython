<h2 ><center  style="color:red;">实干Python笔记</center></h2>



### (壹)科学计算三剑客

#### (一)Numpy常见问题

**[01] 使用Torch进行深度学习任务，要把数组转化张量！**

```python
# Constructing data require the following steps:
# List -> np.array -> Tensor
```

**[02] 留意数组与矩阵的形状是不同的，有可能会误触广播机制！**

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
这种数组与矩阵的维度问题也会出现于torch框架中，
比如下文的经典报错信息:

	UserWarning: Using a target size (torch.Size([16])) that is different to the input size (torch.Size([16, 1])). 
	This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
	
最常用的解决办法是把前向传播的预测值，通过 sequeeze 函数压扁，
如果这个过程已被封装，则在前向传播的函数返回运算结果之前把矩阵压成数组!
"""
```



### (贰) 深度神经网络

#### (一) CNNs

**[01] 只有带可学习参数的模块才算一层**

```python
"""
由于只有Conv与FC带有可学习的参数，
	Conv-ReLU-Conv-ReLU-Pool 
	Conv-ReLU-Conv-ReLU-Pool
	Conv-ReLU-Conv-ReLU-Pool
	FC
因而这个网络结构只有七层,设计网络的时候同样参考这一点,代码如下所示:
"""
class NumBerCNN(nn.Module):
    def __init__(self) -> None:
        super(NumBerCNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(              
                in_channels = 1,            
                out_channels = 16,
                kernel_size = 5,  
                stride = 1,       
                padding = 2       
            ),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size = 2)   # 进行池化操作
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
```



#### (二)RNNs

