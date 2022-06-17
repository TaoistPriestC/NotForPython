# NotForPython

　　用来记录Python语言的学习





### 代码编写命名规范

由于Python语言可能是研究生阶段使用较多的语言，合作编写代码可能是不可避免的，

同时Python语言的变成风格并不像Java那样大体相同，为了交流方便，

本人整理并整合了一套今后将要遵循的命名规范，当然这套规范仅仅是建议，而非强制性的要求：

<table  align="center">
    <caption>命名规范</caption>
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
        <th rowspan="2" align="center" valign="middle">博客信息</th>
        <td colspan="3" align="center" valign="middle">您好，欢迎访问 pan_junbiao的博客</td>
    </tr>
    <tr>
        <td colspan="3" align="center" valign="middle">博客地址：https://blog.csdn.net/pan_junbiao</td>
    </tr>
</table>




| 待命名对象分类 |           样板            |         说明         |
| :------------: | :-----------------------: | :------------------: |
|      类名      |         ClassName         |      大驼峰形式      |
|    常规变量    |         attribute         | 全小写并以下划线分割 |
|      方法      | getAttribute/setAttribute |      小驼峰形式      |
|      函数      |     do_something_work     | 全小写并以下划线分割 |
|      模块      |          module           | 全小写并以下划线分割 |
|    私有方法    |    __do_something_work    |     双下划线开头     |
|    私有变量    |        _attribute         |     单下划线开头     |



### 项目文件夹命名

|    目录名    |          作用           |
| :----------: | :---------------------: |
|     log      |        日志目录         |
|     conf     |        配置目录         |
|     core     |        核心代码         |
| libs/modules |        第三方库         |
|     docs     |         文档库          |
|    README    |        帮助文档         |
|     bin      | 启动入口,存放可执行文件 |
|    tests     |      存放测试代码       |



