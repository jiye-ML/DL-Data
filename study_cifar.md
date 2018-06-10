* 包含60000张32x32的彩色图像， 
* 训练集 50000， 测试集 10000
* cifar10
    * 10类， 每一类图片6000张
    * airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
* CIFAR-100
    * 100 classes
    * 一些类别很小， 例如，包含5种树: maple, oak, palm, pine, and willow. 

* 这里应该存在两种cifar数据，
    * 一种是tensorflow封装过的 record数据， 这种数据后缀都是bin问题
    * 一种是python的可以使用pickle.load 加载的文件，这种文件区别只是没有bin后缀



### 参考文献

* [CNN训练Cifar-10技巧](http://www.cnblogs.com/neopenx/p/4480701.html)



### 各种结果

* alexnet 
    *  a four-layer CNN achieved a 13% test error rate without normalization and 11% with normalization