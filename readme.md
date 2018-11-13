# 机器学习斯坦福大学CS229课程集训营

## 课程资料
1. [课程主页](http://cs229.stanford.edu/)  
2. [中文笔记](https://github.com/learning511/cs229-learning-camp/tree/master/%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%BE%E7%A8%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0)  
3. [课程视频](http://open.163.com/special/opencourse/machinelearning.html)  
4. [作业链接](https://github.com/learning511/cs229-learning-camp/blob/master/assignments.md) 
5. 实验环境推荐使用Linux或者Mac系统，以下环境搭建方法皆适用:  
    [Docker环境配置](https://github.com/ufoym/deepo)  
    [本地环境配置](https://github.com/learning511/cs224n-learning-camp/blob/master/environment.md)


#### 重要一些的资源：
1. [深度学习经典论文](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap.git)
2. [深度学习斯坦福教程](http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)
3. [廖雪峰python3教程](https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000)
4. [github教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
5. [莫烦机器学习教程](https://morvanzhou.github.io/tutorials)
6. [深度学习经典论文](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap.git)
7. [斯坦福cs229代码(机器学习算法python徒手实现)](https://github.com/nsoojin/coursera-ml-py.git)  
8. [吴恩达机器学习新书：machine learning yearning](https://github.com/AcceptedDoge/machine-learning-yearning-cn)  
9. [本人博客(机器学习基础算法专题)](https://blog.csdn.net/dukuku5038/article/details/82253966)  
10. [本人博客(深度学习专题)](https://blog.csdn.net/column/details/28693.html)  


## 前言 
### 这门课的宗旨就是：**“手把手推导机器学习基础理论，一行一行练习徒手代码” ** 

吴恩达在斯坦福的机器学习课，是很多人最初入门机器学习的课，10年有余，目前仍然是最经典的机器学习课程之一。当时因为这门课太火爆，吴恩达不得不弄了个超大的网络课程来授课，结果一不小心从斯坦福火遍全球，而后来的事情大家都知道了。吴恩达这些年，从谷歌大脑项目到创立Coursera再到百度首席科学家再再到最新开设了深度学习deeplearning.ai，辗转多年依然对CS229不离不弃。  

个人认为：吴恩达的cs229的在机器学习入门的贡献相当于牛顿、莱布尼茨对于微积分的贡献。区别在于，吴恩达影响了10年，牛顿影响了200年。

## 数学知识复习  
1.[线性代数](http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf)  
2.[概率论](http://web.stanford.edu/class/cs224n/readings/cs229-prob.pdf)  
3.[凸函数优化](http://web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf)  
4.[随机梯度下降算法](http://cs231n.github.io/optimization-1/)  

#### 中文资料：    
- [机器学习中的数学基本知识](https://www.cnblogs.com/steven-yang/p/6348112.html)  
- [统计学习方法](http://vdisk.weibo.com/s/vfFpMc1YgPOr)  
**大学数学课本（从故纸堆里翻出来^_^）**  

### 编程工具 
#### 斯坦福资料： 
- [Python复习](http://web.stanford.edu/class/cs224n/lectures/python-review.pdf)  
- [TensorFlow教程](https://github.com/open-source-for-science/TensorFlow-Course#why-use-tensorflow)  
#### 中文资料：
- [廖雪峰python3教程](https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000)  
- [莫烦TensorFlow教程](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/)

## 学习安排
每周具体时间划分为4个部分:  
- 1部分安排周一到周二  
- 2部分安排在周四到周五  
- 3部分安排在周日  
- 4部分作业是本周任何时候空余时间    
- 周日晚上提交作业运行截图  
- 周三、周六休息^_^  

#### 作业提交指南：  
 训练营的作业自检系统已经正式上线啦！只需将作业发送到训练营公共邮箱即可，知识星球以打卡为主，不用提交作业。以下为注意事项:  
<0> 课程资料：[链接]() 密码：
<1> 训练营代码公共邮箱：cs229@163.com  
<2> [查询自己成绩:]()  
<3> 将每周作业压缩成zip文件，文件名为“学号+作业编号”，例如："CS229-010037-01.zip"  
<4> 注意不要改变作业中的《方法名》《类名》不然会检测失败！！ 

## 学习安排
### week 1
1.  机器学习的动机与应用 

2.  线性回归、逻辑回归

3. 作业：Assignment 1  
   1.1 Linear Regression  
   1.2 Linear Regression with multiple variables  

### week 2
1.  欠拟合与过拟合的概念 

2.  牛顿方法   

3.  作业：Assignment 2  
   2.1 Logistic Regression  
   2.2 Logistic Regression with Regularization
   
### week 3
1.  生成学习算法 

2.  朴素贝叶斯算法   

3. 作业：Assignment 3  
   3.1 Multiclass Classification  
   
### week 4
1.  最优间隔分类器问题（SVM）

2.  顺序最小优化算法、经验风险最小化   

3. 作业：Assignment 3  
   3.2 Neural Networks Prediction fuction  
   
### week 5
1.  特征选择，神经网络

2.  贝叶斯统计正则化   

3. 作业：Assignment 4  
  4.1 Neural Networks Learning
  
### week 6
1. K-means算法 

2. 高斯混合模型   

3. 作业：Assignment 5  
  5.1 Regularized Linear Regression  
  5.2 Bias vs. Variance  
  
### week 7
1. 主成分分析法（PCA）

2. 奇异值分解（SVD）

3. 作业：Assignment 7  
 7.1 K-means Clustering  
 7.2 Principal Component Analysis  
 
### week 8
1. 马尔可夫决策过程（强化学习初步）

2. 离散与维数灾难 

3. 作业：Assignment 6  
 6.1 Support Vector Machines  
 6.2 Spam email Classifier 
 
 ### week 9
1. 线性二次型调节控制

2. 微分动态规划   

4. 作业：Assignment 8  
 8.1 Anomaly Detection  
 8.2 Recommender Systems  
 
 ### week 10
 1. 策略搜索  
 2. 比赛


  

