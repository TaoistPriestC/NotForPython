## :microscope:项目纲要

这个项目主要用于记录使用Python语言实现的AI学习过程，

接下来给出的提纲主要是对关键部分的梳理，其它内容放在知识库之中，

**机器学习 (Machine Learning)**

- 有监督学习 (Supervised Learning)
- 无监督学习 (Unsupervised Learning)
- 强化学习[^1] (Reinforcement Learning)

**初识神经网络 (Neural Network)**

- 感知机 (Perceptron)

- 深度学习基本问题[^2] (Basic Problem in ML)
  - 回归问题 (Regression)

  - 分类问题 (Classification)
    - 二分类问题 (Binary Classification)

    - 多分类问题 (Multiple Classification)


   - 损失函数 (Cost Function)

        - 反向传播 (Backward)

             - 链式法则 (Chain Rule)
          - 计算图[^3] (Computation Graph)
        - 梯度下降 (Grandient Descent )

             - SGD/MBGD/BGD
             - Gradient with Momentum
             - RMSProp (Root Mean Square Prop)
             - AdaGrad (Adaptive Gradient)
             - Adam (Adaptive Moment Estimation)
- 数学运算相关特性 (Some tricks)

  - 向量化机制 (Vectorization Mechanism)
  - 向量广播机制 (Broadcasting Mechanism)
  - 全连接层间矩阵维度 (Getting Matrix Dimensions Right)

　



## :hammer:代码编写命名规范

由于Python语言科研常用语言，今后合作编写代码难以避免，

同时Python语言的变成风格五花八门，不像Java那样有着一家独尊的趋同之势，

因而为了交流方便，本人整理了一套需要尽量遵循的命名规范，

<table  align="center">
    <tr>
        <th align="center" valign="middle">待命名对象</th>
        <th align="center" valign="middle">样例</th>
        <th align="center" valign="middle">说明</th>
    </tr>
    <tr>
        <td align="center" valign="middle">类名</td>
        <td align="center" valign="middle">ClassName</td>
        <td align="center" valign="middle">大驼峰形式</td>
    </tr>
    <tr>
        <td align="center" valign="middle">常规变量</td>
        <td align="center" valign="middle">attribute</td>
        <td align="center" valign="middle">全小写并以下划线分割</td>
    </tr>
    <tr>
        <td align="center" valign="middle">方法</td>
        <td align="center" valign="middle">getAttribute/setAttribute</td>
        <td align="center" valign="middle">小驼峰形式</td>
    </tr>
    <tr>
        <td align="center" valign="middle">函数</td>
        <td align="center" valign="middle">do_something_work</td>
        <td align="center" valign="middle">全小写并以下划线分割</td>
    </tr>
     <tr>
        <td align="center" valign="middle">模块</td>
        <td align="center" valign="middle">module</td>
        <td align="center" valign="middle">全小写并以下划线分割</td>
    </tr>
    <tr>
        <td align="center" valign="middle">函数</td>
        <td align="center" valign="middle">__do_something_work</td>
        <td align="center" valign="middle">双下划线开头</td>
    </tr>
    <tr>
        <td align="center" valign="middle">私有变量</td>
        <td align="center" valign="middle">_attribute</td>
        <td align="center" valign="middle">单下划线开头</td>
    </tr>
    <tr>
        <th rowspan="2" align="center" valign="middle">项目文件夹命名</th>
        <td colspan="3" align="center" valign="middle">日志目录 - 配置目录 - 核心代码 - 第三方库 - 文档库 - 帮助文档 - 启动入口 - 测试代码</td>
    </tr>
    <tr>
        <td colspan="3" align="center" valign="middle">log - conf - core - libs/modules - docs - README - bin - tests</td>
    </tr>
</table>

当然规范仅仅是建议，而非强制性的要求，你可以不遵守，如果你可以写出更加优化的代码！







　

## :page_with_curl:文中的批注

[^1]: 强化学习与监督学习类似，但在学习过程中没有标签提示对错，而是通过自我评估的方法进行调整，其在机器人控制、计算机游戏、市场战略等领域有着广泛应用，强化学习会用到回归、分类、聚类、降维等各种各样机器学习的算法。
[^2]: 深度学习<表示学习<机器学习。目前来说，深度学习的三大核心包括了数据、计算资源以及算法，通常有一个新想法，需要写出代码，然后选择一批数据集验证，如果任务规模较大可能需要挂在服务器跑一个月甚至更久。

[^3]: 对于国内的学生来说，手算链式求导几乎没有任何压力，但是计算图仍然有必要深入学习，因为这是Pytorch自动求导机制的实现基础。

