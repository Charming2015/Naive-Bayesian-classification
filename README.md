# Naive-Bayesian-classification
### javascript实现朴素贝叶斯分类器（Naive-Bayesian-classification）
#### 不懂啥是贝叶斯分类器的可以看这两篇文章
1. [朴素贝叶斯分类器的应用](http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html)
2. [算法杂货铺——分类算法之朴素贝叶斯分类(Naive Bayesian classification)](http://www.cnblogs.com/leoo2sk/archive/2010/09/17/naive-bayesian-classifier.html)

-----
- 这个函数不用全看懂，只需要知道这个是用来计算数组的均值和方差就行
- 这个一开始是写着玩的，进行普通的统计描述。|
- calcRepeat是用来计算数组的重复值的
````javascript
var StatisticalDesc = function(arr , num){
	var newArr = new Array;
	for(var i in arr){
		newArr[i] = arr[i]
	}
	var arrLen = newArr.length;
	var total = 0,
		mean = 0,  //均值
		variance = 0,  //方差
		sd = 0,  // 标准差
		mode = 0,  //众数
		median = 0; //中位数
	// 计算均值
	for(var i in newArr ){
		total += newArr[i]
	}
	mean = total / arrLen;

	/*
	计算方差
	注意估计方差的方法
	http://baike.baidu.com/item/%E6%96%B9%E5%B7%AE	
	*/
	for(var i in newArr){
		variance += (newArr[i]-mean)*(newArr[i]-mean)
	}

	/*
	关于除以n还是除以n-1
	如是总体（即估算总体方差），根号内除以n
	如是抽样（即估算样本方差），根号内除以（n-1）
	*/
	variance = variance.toFixed(6) / (arrLen-1)
	// 计算标准差
	sd = Math.sqrt(variance)
	// 计算中位数
	newArr.sort(function(a,b){
		return a-b
	})
	if(arrLen % 2 == 0){
		median = (newArr[arrLen/2] + newArr[arrLen/2-1]) / 2;
	}else{
		median = newArr[Math.floor(arrLen/2)]
	}
	// 计算众数
	var maxTimes = 0;
	for(var i in newArr){
		var temp = this.calcRepeat(newArr,newArr[i]);
		if(maxTimes < temp){
			maxTimes = temp;
			mode = newArr[i];
		}
	}
	if(maxTimes == 1){
		mode = 'none'
	}
	// 保留小数位
	if(num) {
		sd = sd.toFixed(num)
		mean = mean.toFixed(num)
		variance = variance.toFixed(num)
	};
	return {mean:mean,variance : variance , sd:sd , median:median, mode:mode};
}
StatisticalDesc.prototype.calcRepeat = function(arr , value){
	var times = 0;
	for(var i in arr ){
		if(arr[i] == value){
			times ++
		}
	}
	return times;
}
````
- 造一个Bayes类
- type_1_Arr和type_2_Arr是用来储存数据，按照type的分开的类的数据
- obj是想检测（想归类）的对象
- 我把data的第一个type作为第一类型
````javascript
var Bayes = function(data,obj){
	this.data = data;
	this.type_1_Arr = [];
	this.type_2_Arr = [];
	this.obj = obj;
	this.type_1 = this.data[0].type;
	this.type_2 = '';
}
````
- 按照type把数据（data）分开类
````javascript
Bayes.prototype.class = function(){
	for(var i in this.data){
		if(this.data[i].type == this.type_1){
			this.type_1_Arr.push( this.data[i] )
		}else{
			this.type_2_Arr.push( this.data[i] )
			this.type_2 = this.data[i].type
		}
	}
}
````
- 计算每个特征的概率（正态分布），这里我全部默认数据服从正态分布
````javascript
Bayes.prototype.norDis = function( x , mean , variance){
	var deno = Math.sqrt(2 * Math.PI * variance );
	var exp = -(x-mean)*(x-mean)/(2* variance )
	return Math.pow(Math.E , exp)/deno
}
````
- 这个函数可能有点难看懂，因为是我写好之后再提取公共部分形成的。
- 根据这个公式：**P(身高=6|男) x P(体重=130|男) x P(脚掌=8|男) x P(男)**，对照着看应该能懂
	- p_temp是用来把*P(身高=6|男) x P(体重=130|男) x P(脚掌=8|男) *连乘起来的。（因为以后可能会有4个，5个更多的特征）
	- p_type是 P(男)的意思
- allArr的是用来存储除了type之外的数据，格式大概这样子
	````javascript
	[
	{'特征1':[]},
	{'特征2':[]},
	{'特征3':[]},
	.....
	]
	````

- 第一个for循环是把allArr给格式化
- 第二个for循环是把allArr的数据填好（至于里面为啥还要有一个循环呢？因为参数的名字和个数我都不知道，只能根据数组自己的key来）
- 这样说应该懂了吧
````javascript
Bayes.prototype.calcP = function(arr,obj){
	var allArr = [];
	for(var i in arr[0]){
		if( i != 'type'){
			allArr[i] = [];
		}
	}
	for(var i in arr){
		for(k in arr[0]){
			if( k != 'type'){
				allArr[k].push(arr[i][k])
			}
		}
	}	
	var p_temp = 1;
	for(var i in allArr){
		var temp_statDesc = new StatisticalDesc(allArr[i] , 3);
		p_temp *= this.norDis(this.obj[i],temp_statDesc.mean,temp_statDesc.variance)
	}

	var p_type = arr.length / this.data.length
	var p_guess = p_temp * p_type

	return p_guess;
}
````
- 程序的主逻辑
- 首先把数据按照type分类
- 然后算好预测数据属于两个类型的概率，然后比较概率
````javascript
Bayes.prototype.init = function(){
	this.class();
	var p_type_1 = this.calcP(this.type_1_Arr) 
	var p_type_2 = this.calcP(this.type_2_Arr) 
	if(p_type_1 > p_type_2){
		console.log('猜测为'+this.type_1,parseInt(p_type_1/p_type_2))
		return '猜测为'+this.type_1+'  概率倍数：' + parseInt(p_type_1/p_type_2)
	}else{
		console.log('猜测为'+this.type_2,parseInt(p_type_2/p_type_1))
		return '猜测为'+ this.type_2 +'  概率倍数：' + parseInt(p_type_2/p_type_1)
	}
}
````
#### 不懂的童鞋可以自行下载代码研究
