### 1.  加L1与L2正则化的线性回归

```python
from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("LinearRegressionWithElasticNet")\
    .getOrCreate()

# 加载数据
training = spark.read.format("libsvm")\
    .load("../data/mllib/sample_linear_regression_data.txt")

# 初始化模型参数（迭代10轮，正则化强度0.3，加了L1与L2正则化的线性回归叫elasticNetPara=0.8）  
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8) 

# 拟合模型
lrModel = lr.fit(training)

# 输出系数和截距 y=WX+b
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# 模型信息总结输出
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

spark.stop()
```

> ```latex
> Coefficients: [0.0,0.32292516677405936,-0.3438548034562218,1.9156017023458414,0.05288058680386263,0.765962720459771,0.0,-0.15105392669186682,-0.21587930360904642,0.22025369188813426]
> Intercept: 0.1598936844239736
> numIterations: 7
> objectiveHistory: [0.49999999999999994, 0.4967620357443381, 0.4936361664340463, 0.4936351537897608, 0.4936351214177871, 0.49363512062528014, 0.4936351206216114]
> +--------------------+
> |           residuals|
> +--------------------+
> |  -9.889232683103197|
> |  0.5533794340053554|
> |  -5.204019455758823|
> | -20.566686715507508|
> |    -9.4497405180564|
> |  -6.909112502719486|
> |  -10.00431602969873|
> |   2.062397807050484|
> |  3.1117508432954772|
> | -15.893608229419382|
> |  -5.036284254673026|
> |   6.483215876994333|
> |  12.429497299109002|
> |  -20.32003219007654|
> | -2.0049838218725005|
> | -17.867901734183793|
> |   7.646455887420495|
> | -2.2653482182417406|
> |-0.10308920436195645|
> |  -1.380034070385301|
> +--------------------+
> only showing top 20 rows
>
> RMSE: 10.189077
> r2: 0.022861
> ```

### 2.  广义线性模型

```python
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression


spark = SparkSession\
    .builder\
    .appName("GeneralizedLinearRegressionExample")\
    .getOrCreate()

# 加载数据
dataset = spark.read.format("libsvm")\
    .load("../data/mllib/sample_linear_regression_data.txt")

# 初始化模型参数  
glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# 拟合模型
model = glr.fit(dataset)

# 输出系数和截距
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# 模型信息总结与输出
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()

spark.stop()
```

> ```latex
> Coefficients: [0.010541828081257216,0.8003253100560949,-0.7845165541420371,2.3679887171421914,0.5010002089857577,1.1222351159753026,-0.2926824398623296,-0.49837174323213035,-0.6035797180675657,0.6725550067187461]
> Intercept: 0.14592176145232041
> Coefficient Standard Errors: [0.7950428434287478, 0.8049713176546897, 0.7975916824772489, 0.8312649247659919, 0.7945436200517938, 0.8118992572197593, 0.7919506385542777, 0.7973378214726764, 0.8300714999626418, 0.7771333489686802, 0.463930109648428]
> T Values: [0.013259446542269243, 0.9942283563442594, -0.9836067393599172, 2.848657084633759, 0.6305509179635714, 1.382234441029355, -0.3695715687490668, -0.6250446546128238, -0.7271418403049983, 0.8654306337661122, 0.31453393176593286]
> P Values: [0.989426199114056, 0.32060241580811044, 0.3257943227369877, 0.004575078538306521, 0.5286281628105467, 0.16752945248679119, 0.7118614002322872, 0.5322327097421431, 0.467486325282384, 0.3872259825794293, 0.753249430501097]
> Dispersion: 105.60988356821714
> Null Deviance: 53229.3654338832
> Residual Degree Of Freedom Null: 500
> Deviance: 51748.8429484264
> Residual Degree Of Freedom: 490
> AIC: 3769.1895871765314
> Deviance Residuals: 
> +-------------------+
> |  devianceResiduals|
> +-------------------+
> |-10.974359174246889|
> | 0.8872320138420559|
> | -4.596541837478908|
> |-20.411667435019638|
> |-10.270419345342642|
> |-6.0156058956799905|
> |-10.663939415849267|
> | 2.1153960525024713|
> | 3.9807132379137675|
> |-17.225218272069533|
> | -4.611647633532147|
> | 6.4176669407698546|
> | 11.407137945300537|
> | -20.70176540467664|
> | -2.683748540510967|
> |-16.755494794232536|
> |  8.154668342638725|
> |-1.4355057987358848|
> |-0.6435058688185704|
> |  -1.13802589316832|
> +-------------------+
> only showing top 20 rows
> ```

### 3.  逻辑回归

```python
from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("LogisticRegressionSummary") \
    .getOrCreate()

# 加载数据
training = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 拟合模型
lrModel = lr.fit(training)

# 模型信息总结与输出
trainingSummary = lrModel.summary

# 输出每一轮的损失函数值
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# ROC曲线
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
#fMeasure = trainingSummary.fMeasureByThreshold
#maxFMeasure = fMeasure.groupBy(['threshold']).max('F-Measure').select('max(F-Measure)')
#bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure.select('max(F-Measure)')['max(F-Measure)']).select('threshold')['threshold']
#lr.setThreshold(bestThreshold)

spark.stop()
```

> ```latex
> objectiveHistory:
> 0.6833149135741672
> 0.6662875751473734
> 0.6217068546034618
> 0.6127265245887887
> 0.6060347986802873
> 0.6031750687571562
> 0.5969621534836274
> 0.5940743031983118
> 0.5906089243339022
> 0.5894724576491042
> 0.5882187775729587
> +---+--------------------+
> |FPR|                 TPR|
> +---+--------------------+
> |0.0|                 0.0|
> |0.0|0.017543859649122806|
> |0.0| 0.03508771929824561|
> |0.0| 0.05263157894736842|
> |0.0| 0.07017543859649122|
> |0.0| 0.08771929824561403|
> |0.0| 0.10526315789473684|
> |0.0| 0.12280701754385964|
> |0.0| 0.14035087719298245|
> |0.0| 0.15789473684210525|
> |0.0| 0.17543859649122806|
> |0.0| 0.19298245614035087|
> |0.0| 0.21052631578947367|
> |0.0| 0.22807017543859648|
> |0.0| 0.24561403508771928|
> |0.0|  0.2631578947368421|
> |0.0|  0.2807017543859649|
> |0.0|  0.2982456140350877|
> |0.0|  0.3157894736842105|
> |0.0|  0.3333333333333333|
> +---+--------------------+
> only showing top 20 rows
>
> areaUnderROC: 1.0
> ```

