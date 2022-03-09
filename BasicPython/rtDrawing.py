import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ax = []   # 用于保存图1数据
ay = []
bx = []   # 用于保存图2数据
by = []

"""  
  matplotlib配置文件也即.rc文件, 其为输出图形的几乎所有属性指定了默认值,
如果不喜欢其提供的默认属性, 则可使用参数字典 rcParams 访问并修改已经加载的配置项!
  
  我们有时候需要实时交互功能, 这项工具需要记住以下几个常用的函数
      - 开启交互模式 - plt.ion()  
      - 清除图片缓存 - plt.clf()
      - 关闭交互模式 - plt.ioff()
      - 设置暂停时间 - plt.pause()
      - 绘制并且显示 - plt.show() 
"""

plt.ion()    
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 0.5  


#=======================================================================#
# 设置随机数实时显示区域的文字参数
RT_XLOC = 0.20
RT_YLOC = 0.95
FONT_DICT  = dict(fontsize = 12, color = "r", family="Microsoft YaHei")
BGBOX = {
    "facecolor": "#FFFFE0", # 填充色
    'edgecolor':'r',        # 外框色
    'alpha': 0.5,           # 框透明度
    'pad': 8,               # 本文与框周围距离 
}
# 关于文字区域背景色设置的方法参考了下面这篇文章:
# https://zhuanlan.zhihu.com/p/205110001
# 关于下文随机数采样设置的方法参考了下面这篇文章:
# https://blog.csdn.net/ddjhpxs/article/details/107766742
#=======================================================================#



# 通过的循环生成一百组案例并实时绘制
for i in range(100):
    plt.clf()    
    plt.suptitle("Title", fontsize=30)
	
    # 下面的代码使用了plt.subplot(rows, cols, idx)分别的创建了两张子图
    # 共有子图张数: rows × cols, 其中idx = 1, 2, 3, ... , row × cols
    # 代表子图编号, 函数会返回 "AxesSubplot" 子图对象, 
    # 可直接通过子图对象进行操作,


    # subplot-1
    g1 = np.random.random()
    ax.append(i)        
    ay.append(g1)       
    agraphic = plt.subplot(1, 2, 1)
    agraphic.set_title("Subplot1")
    agraphic.text(RT_XLOC, RT_YLOC, "{:.3f}".format(g1)
        , transform = plt.gca().transAxes
        , fontdict = FONT_DICT
        , bbox = BGBOX
    )
    agraphic.set_ylim(0, 1.5)
    agraphic.set_xlabel(r"$X$", fontsize = 15) 
    agraphic.set_ylabel(r"$y$", fontsize = 15)    
    agraphic.plot(ax,ay,"r-")

    # subplot-2
    g2 = np.random.random()
    bx.append(i)
    by.append(g2)
    bgraphic = plt.subplot(1, 2, 2)
    bgraphic.set_title("Subplot2")
    bgraphic.text(RT_XLOC, RT_YLOC, "{:.3f}".format(g2)
        , transform = plt.gca().transAxes
        , fontdict = FONT_DICT
        , bbox = BGBOX
    )
    bgraphic.set_ylim(0, 1.5)
    bgraphic.set_xlabel(r"$X$", fontsize = 15)
    bgraphic.set_ylabel(r"$y$", fontsize = 15)    
    bgraphic.plot(bx,by,"b-")

    ## 上面的g1与g2用于生成随机数, 但是生成随机数的函数其实很多, 根据自己的需要可以对其进行改造,
    ## 我们希望把这个数字显示到子图上面, 使用text函数, 并且设置参数transform, 
    ## 随后前面两个参数将会变成相对坐标, 否则将会使用坐标系的绝对坐标,
    plt.pause(0.01)

plt.ioff()      
plt.show()      

# 由于程序需要启动系统调用time.sleep(interval) 暂停 interval (s)
# 所以这个程序其实运行起来是比较缓慢的, 想要提前终止可在终端按下 CTRL+C 或 CTRL+Z, 不同系统略有差别