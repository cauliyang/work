# K-means

K-means 聚类 Python实现


**本篇文章记录学习过程中的点滴-K-means聚类方法，在命令行端运行，Python手写算法步骤。添加递归次数阈值。如果有什么问题和可以改进的地方欢迎提出问题，共同进步～**

__脚本用法__:python 脚本名 k值 原始数据文件 最大迭代次数 初始质心文件(可选)
  ___注意___:
-    1.若要提供质心文件则K值需与提供质心数量一致.
-    2.若使用Windows运行可能会出现编码问题建议在mac,linux系统下运行。
-    3.结果文件名为kmean.out
-    4.需要Numpy模块


__算法思路__:
-    1.获取所需信息原始数据文件,初始化质心.
-    2.计算所有数据到质心的欧式距离,根据最小距离分群.
-    3.重新根据分群结果计算质心,并比较质心是否变化,以及是否到达最大迭代次数.
-    4.若有变化且没有达到最大迭代次数，则重复2 , 3 两步,直到达到最大迭代数或者质心不发生变化迭代停止进行下一步.
-    5.输出结果并生成报告.



-------------------------------------------------
__Requirement : numpy__

__Usage : python k_mean.py k data max_it (cetroids)__

__Result_file : kmeans.out__

-------------------------------------------------

