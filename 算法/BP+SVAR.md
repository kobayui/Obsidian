# 采用pytorchgpu训练时间更长

## 第一种设置

~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchsize = 32

    lags = 2

    features = 6

    learning_rate = 0.005

    epochs = 30

    a = 20000

    b = 20334  # b - train_size > lags

    train_size = 322 # batch = train_size - lags  /  batchsize

    train_loss=[]

    train_loss_epoch = []

    train_x, train_y, test_x, test_y, mm = load_data(a, b)
~~~

同样的设置：

### pytorchgpu结果：109.8s

准确率：
5.775455576957144e-06
5.543061855915372e-05
0.00018024798851746233
0.0013344971071426643
0.0017403247360750784
0.0010702758302027888
Running time: 10.875713658332824 Seconds
训练误差：
0.007284352498526762
r2：
0.9345733234116989
0.921298008419822
0.9426341799876365
0.9193959705901953
0.940559138622412
0.9523754775352511

### 自写结果：1.9s

准确率：
6.767347371190517e-05
0.0005838555608676714
0.0007720416390234232
0.0029476908375489492
0.008608539320562384
0.006690007786022465
Running time: 0.1881129503250122 Seconds
训练误差：
0.02017138739358764
r2：
-0.013943392674505117
-0.11757469721397766
0.1766082706767533
-0.08756190077833446
-1.191984739970712
-0.05084470232296704

## 第二种设置

~~~
batchsize = 64

    lags = 2

    features = 6

    learning_rate = 0.005

    epochs = 40

    a = 20000
    
    b = 20649  # b - train_size > lags
    
    train_size = 642 # batch = train_size - lags  /  batchsize
~~~

### pytorchgpu结果：152.4s

准确率：
3.884961031987743e-06
2.1311293267203507e-05
0.0001277236259001877
0.000512737878083335
0.0038758512162744446
0.0008114363983899153
Running time: 30.22213625907898 Seconds
训练误差：
0.004330030679446281
r2：
0.9968562768064848
0.9937707670128532
0.9891454319768828
0.9845980475548555
-0.6304671768411769
0.9797869228431741

***learning_rate=0.01***
准确率：
9.462532297898346e-06
9.062859310575811e-05
6.146935142950593e-05
0.0006020762904876138
0.0004106672527945188
0.0007461077586618549
Running time: 29.848733615875243 Seconds
训练误差：
0.006111101716815029
r2：
0.9860569585552239
0.9894792481080541
0.9965729030984616
0.8993572649084962
0.9940786311431167
0.9936394442683479

***learning_rate=0.01***  ***epochs=20***
准确率：
7.661263542806947e-06
8.975929373084465e-05
0.00019199296768731384
0.0006198082816918969
0.0006108824728034781
0.0003878211261912968
Running time: 16.166567611694337 Seconds
训练误差：
0.010671779709271504
r2：
0.8857780374121494
0.7510172259990984
0.9631487874234044
0.9851747957003818
0.9879684503030604
0.9927557661499146


### 自写结果：2.6s

准确率：
0.0002406152602199208
0.0016213718405145392
0.004838433155987575
0.009762625882833319
0.05621290124709437
0.016972436947269857
Running time: 0.5178226470947266 Seconds
训练误差：
0.15085552981006511
r2：
-6.798121544932856
-8.360548993973234
-4.020920509076289
-6.901269445897104
-7.3645689757265185
-6.383891160934918

## 第三种

~~~
batchsize = 12

    lags = 2

    features = 6

    learning_rate = 0.01

    epochs = 20

    a = 20000

    b = 20129  # b - train_size > lags

    train_size = 122 # batch = train_size - lags  /  batchsize
~~~

### pytorchgpu结果：17.8s

准确率：
3.389847689778381e-05
9.41134248487635e-05
0.00032067952027349945
0.0007009946579031947
0.0006598316948875221
0.0014404306058395005
Running time: 3.5098053932189943 Seconds
训练误差：
0.030840756284218514
r2：
0.9006715413861979
0.9251281671652792
0.8056835809289504
0.9863441205041713
0.9799025429416295
0.9457928150330333

***learning_rate=0.01***  ***epochs=20***    17.5s

准确率：
2.6711665237128257e-05
0.000164000322576005
0.0002268064028474645
0.0022165381986196985
0.003989689531835846
0.0011502986270056556
Running time: 3.4441831588745115 Seconds
训练误差：
0.02841923930525081
r2：
0.8253678973267808
0.8630200312104156
0.8978232712195172
0.7707906952971628
0.9176305350948336
0.917602195834865

## 第四种

~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchsize = 2

    lags = 2

    features = 6

    learning_rate = 0.005

    epochs = 20

    a = 20000

    b = 20129  # b - train_size > lags

    train_size = 122 # batch = train_size - lags  /  batchsize
~~~

### gpu结果

准确率：
1.904527214371998e-05
0.00010116005744525672
0.0002386281093022624
0.0004548408067897822
0.004446547488925592
0.0006062007285518258
Running time: 5.245997381210327 Seconds
训练误差：
0.011530233424460069
r2：
0.8399189694573306
0.8017388112073263
0.9299239368684429
0.9659608441660683
0.7508693271856911
0.9790587100039717

### CPU结果

准确率：
1.2886932110756833e-05
0.00015751779480143406
0.00018391139702273447
0.00039496802133605016
0.003685535941050584
0.0021096788043743573
Running time: 7.220826911926269 Seconds
训练误差：
0.011815238463428007
r2：
0.9314112448155333
0.73625025551509
0.9275725700277999
0.9555313061483292
0.8693155421743229
0.8194549485830235

## 第五种

~~~
device = torch.device("cpu")

    batchsize = 90

    lags = 60

    features = 6

    learning_rate = 0.01

    epochs = 20

    a = 20000

    b = 21921  # b - train_size > lags

    train_size = 1860 # batch = train_size - lags  /  batchsize
~~~

### 新-cpu

准确率：
1.789311423367293e-05
0.012146952894703258
0.0020894123586100892
0.0596878361690468
0.09827739432510044
0.07848929153728916
Running time: 982.90913438797 Seconds
训练误差：
9.959467064086349
r2：
-0.8439801719450077
-541.0889755676914
-2.8588249042688276
-124.21442026635677
-86.04901398546956
-19.16813007916767

### 旧-cpu


---------------------------------------------------------------------------
~~~
ValueError                                Traceback (most recent call last)
Cell In[2], line 442
    440 test_x = train_data[:-number,:]
    441 test_x = train_data
--> 442 accuracy, res, testy,r2,z2 = test_online(test_x, lags=lags)
    443 # accuracy, res, testy = test_online(test_data, lags=lags)
    444 # print(accuracy)
    446 r2_list_1.append(r2[0])

Cell In[2], line 296, in test_online(test_x, lags)
    294 for i in range(col):
    295     accuracy[i]=get_mape(test_y[-1,i], res[-1,i])
--> 296     r2[i]=get_r2(test_y[:,i], res[:,i])
    298 return accuracy, res, test_y,r2,z2

Cell In[2], line 235, in get_r2(x, y)
    234 def get_r2(x,y):
--> 235     return r2_score(y_true=x, y_pred=y)

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\utils\_param_validation.py:218, in validate_params.<locals>.decorator.<locals>.wrapper(*args, **kwargs)
    212 try:
    213     with config_context(
    214         skip_parameter_validation=(
    215             prefer_skip_nested_validation or global_skip_validation
    216         )
    217     ):
--> 218         return func(*args, **kwargs)
    219 except InvalidParameterError as e:
    220     # When the function is just a wrapper around an estimator, we allow
    221     # the function to delegate validation to the estimator, but we replace
    222     # the name of the estimator by the name of the function in the error
    223     # message to avoid confusion.
    224     msg = re.sub(
    225         r"parameter of \w+ must be",
    226         f"parameter of {func.__qualname__} must be",
    227         str(e),
    228     )

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\metrics\_regression.py:1276, in r2_score(y_true, y_pred, sample_weight, multioutput, force_finite)
   1152 """:math:`R^2` (coefficient of determination) regression score function.
   1153 
   1154 Best possible score is 1.0 and it can be negative (because the
   (...)   1269 -inf
   1270 """
   1271 xp, _, device_ = get_namespace_and_device(
   1272     y_true, y_pred, sample_weight, multioutput
   1273 )
   1275 _, y_true, y_pred, sample_weight, multioutput = (
-> 1276     _check_reg_targets_with_floating_dtype(
   1277         y_true, y_pred, sample_weight, multioutput, xp=xp
   1278     )
   1279 )
   1281 if _num_samples(y_pred) < 2:
   1282     msg = "R^2 score is not well-defined with less than two samples."

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\metrics\_regression.py:209, in _check_reg_targets_with_floating_dtype(y_true, y_pred, sample_weight, multioutput, xp)
    160 """Ensures y_true, y_pred, and sample_weight correspond to same regression task.
    161 
    162 Extends `_check_reg_targets` by automatically selecting a suitable floating-point
   (...)    205     correct keyword.
    206 """
    207 dtype_name = _find_matching_floating_dtype(y_true, y_pred, sample_weight, xp=xp)
--> 209 y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
    210     y_true, y_pred, sample_weight, multioutput, dtype=dtype_name, xp=xp
    211 )
    213 return y_type, y_true, y_pred, sample_weight, multioutput

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\metrics\_regression.py:116, in _check_reg_targets(y_true, y_pred, sample_weight, multioutput, dtype, xp)
    114 check_consistent_length(y_true, y_pred, sample_weight)
    115 y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
--> 116 y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    117 if sample_weight is not None:
    118     sample_weight = _check_sample_weight(sample_weight, y_true, dtype=dtype)

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\utils\validation.py:1105, in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_non_negative, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
   1099     raise ValueError(
   1100         f"Found array with dim {array.ndim},"
   1101         f" while dim <= 2 is required{context}.
   1102     )
   1104 if ensure_all_finite:
-> 1105     _assert_all_finite(
   1106         array,
   1107         input_name=input_name,
   1108         estimator_name=estimator_name,
   1109         allow_nan=ensure_all_finite == "allow-nan",
   1110     )
   1112 if copy:
   1113     if _is_numpy_namespace(xp):
   1114         # only make a copy if `array` and `array_orig` may share memory`

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\utils\validation.py:120, in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    117 if first_pass_isfinite:
    118     return
--> 120 _assert_all_finite_element_wise(
    121     X,
    122     xp=xp,
    123     allow_nan=allow_nan,
    124     msg_dtype=msg_dtype,
    125     estimator_name=estimator_name,
    126     input_name=input_name,
    127 )

File c:\Users\Lenovo\.conda\envs\pytorchgpu\Lib\site-packages\sklearn\utils\validation.py:169, in _assert_all_finite_element_wise(X, xp, allow_nan, msg_dtype, estimator_name, input_name)
    152 if estimator_name and input_name == "X" and has_nan_error:
    153     # Improve the error message on how to handle missing values in
    154     # scikit-learn.
    155     msg_err += (
    156         f"\n{estimator_name} does not accept missing values"
    157         " encoded as NaN natively. For supervised learning, you might want"
   (...)    167         "#estimators-that-handle-nan-values"
    168     )
--> 169 raise ValueError(msg_err)

ValueError: Input contains NaN.
~~~

## 第六种

~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    batchsize = 2
    lags = 2
    features = 6
    learning_rate = 0.005
    epochs = 50

    a = 20000
    b = 20186  # b - train_size > lags

    train_size = 182
~~~

### GPU-36.7s

~~~
准确率：
3.82887304354412e-06
6.892516923956782e-05
0.0001774751649253885
0.0005698415384950723
0.005418637962095711
0.001477755734955715
Running time: 17.1111980676651 Seconds
训练误差：
0.011058592317749569
r2：
0.9527703642840615
0.9451631728158387
0.92150507518533
0.9639569069863136
0.9202372003236929
0.891326851879945
~~~

### CPU-新-11.4s

~~~
准确率：
2.564797032644842e-06
1.0703622824295754e-05
0.00019333993010343348
0.00010370393711457652
0.0016550806321832627
0.0004894478366674131
Running time: 5.685866832733154 Seconds
训练误差：
0.009027551005442749
r2：
0.995879456266885
0.9878299721106671
0.9468048420326609
0.9846532256771716
0.9398049103746895
0.9770608616758311
~~~

### CPU-旧-3.1s

~~~
准确率：
2.7155238695017355e-05
0.00013500328615039065
0.0003235712380798884
0.0006914659599955465
0.0017504866437058665
0.0029174462102501686
Running time: 0.3202592134475708 Seconds
训练误差：
0.11989456476773538
r2：
0.9157732219474324
0.9218929023020785
0.9406599425310519
0.8345706264163417
0.8715588517357364
0.8789388782841214
~~~

## 第七种

~~~
device = torch.device("cpu")
    batchsize = 20
    lags = 2
    features = 6
    learning_rate = 0.005
    epochs = 50

    a = 20000
    b = 20186  # b - train_size > lags

    train_size = 182
~~~

### cpu-新-4.9s

~~~
准确率：
9.29536796270141e-06
0.00015673177167086175
0.00029223318981694074
0.0016635338876492707
0.002840260153487039
0.001650857843353552
Running time: 2.442058563232422 Seconds
训练误差：
0.0212470121016506
r2：
0.9203548388938005
0.991303883203584
0.9865491356290921
0.9216365406599877
0.9529250235376393
0.8874005475886916
~~~

### gpu-19.3s

~~~
准确率：
6.256331198232309e-06
0.0001340740182849898
0.0006278487635212554
0.0004146005387262439
0.001871558479788254
0.0015385332637001983
Running time: 9.608861088752747 Seconds
训练误差：
0.019868787920471122
r2：
0.99633548527267
0.982989016809943
0.7069616285139395
0.9582893948245672
0.98971663767663
0.9794977045596149
~~~

### CPU-旧-0.4s

~~~
多次试验得到比较好的结果
准确率：
8.683450594773752e-05
0.0009294412116260575
0.001887511407128672
0.004629198806423213
0.010506891173812967
0.011576569166146506
Running time: 0.16900551319122314 Seconds
训练误差：
0.138537170080693
r2：
0.6652843485481494
0.2553104133463122
0.6894705841291995
0.4844986640392649
0.7522934486484357
0.2975238583863506
~~~

## 第八种

~~~
device = torch.device("cpu")
    batchsize = 30
    lags = 2
    features = 6
    learning_rate = 0.005
    epochs = 50

    a = 20000
    b = 20606  # b - train_size > lags

    train_size = 602 # batch = train_size - lags  /  batchsize
~~~
### CPU-新-16.1s

~~~
准确率：
5.509681910099737e-06
5.7413336524088016e-05
0.0002201882436386028
0.00015016607453063473
0.0009521003942806886
0.000331083337219269
Running time: 7.990940093994141 Seconds
训练误差：
0.004935505016248499
r2：
0.9916221948505204
0.9939660476798131
0.9238818139076502
0.9482227005785999
0.946707528842671
0.9689841052228612
~~~

### gpu-64.3s

~~~
准确率：
4.1046360587885925e-06
2.1890360615443115e-05
0.00011645441089044167
0.0008969584081235959
0.0002767475678922702
0.001036981199328348
Running time: 31.991248726844788 Seconds
训练误差：
0.004528281647053519
r2：
0.9971278563502213
0.9629708152544583
0.9959413601078391
0.9458839422444913
0.9292501389286068
0.9329831254140768
~~~

## 第九种

~~~
device = torch.device("cpu")

    batchsize = 60

    lags = 2

    features = 6

    learning_rate = 0.005

    epochs = 30

    a = 20000

    b = 20606  # b - train_size > lags

    train_size = 602
~~~
### cpu-新-9.6s

~~~
多次试验，否则r2会出现负数情况
准确率：
2.8334199901467444e-06
5.186652937590841e-05
0.0001511979755908776
0.0008438632753315211
0.0003446057988908544
0.0007126195576497193
Running time: 4.769944906234741 Seconds
训练误差：
0.008009062825161286
r2：
0.996131428111684
0.9852553440014178
0.9963754174667705
0.9957446092575368
0.9915924310161818
0.9952853798331016
~~~

### gpu

~~~
准确率：
1.288919988792008e-05
8.169669928593447e-05
0.00014673107637742052
9.056855724795358e-05
0.0007325258464924417
0.0007185791573401485
Running time: 18.895015835762024 Seconds
训练误差：
0.00849459965240385
r2：
0.9405845846175297
0.9443101988970255
0.9937134051833265
0.9931402413758583
0.9844530901187477
0.9933508090879015
~~~

## 总结

1. cpu-新比使用gpu训练时间更快
2. 训练结果中很可能会出现r2为负数的情况
3. 对于lags的选择可使用bic准确预先确定好范围，在试验中发现增加lags并不会让结果变得更好
4. ~~~
   device = torch.device("cpu")
    batchsize = 10
    lags = 2
    features = 6
    learning_rate = 0.005
    epochs = 30
    a = 20000
    b = 20186  # b - train_size > lags
    train_size = 182
   ~~~
5. 结果
~~~
准确率：
3.0357321602224895e-06
2.208586922203867e-05
8.13533099616724e-05
0.0012098994104395564
0.0007776717544407082
0.0006813860820042327
Running time: 1.691620111465454 Seconds
训练误差：
0.01471438540119615
r2：
0.9970605695116741
0.9910750579748815
0.9944885480692381
0.9705986671866333
0.9863352294875544
0.9846667463585423
~~~

6. 训练误差不要看平均值，看最后的结果或是趋势
7. 代码

~~~
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import r2_score

class BP(nn.Module):
    def __init__(self, batchsize, lags, features, learning_rate):
        super(BP, self).__init__()
        
        self.batchsize = batchsize
        self.lags = lags
        self.features = features
        self.num = batchsize
        self.outnum = self.features * self.features * (self.lags + 1)
        self.hidnum = self.features * self.features
        self.learning_rate = learning_rate

        # 线性层：输入到隐藏层
        self.fc1 = nn.Linear(self.features, self.hidnum)
        # 线性层：隐藏层到输出层
        self.fc2 = nn.Linear(self.hidnum, self.outnum)

    def forward(self, x, h):
        # 前向传播
        hidden_layer_1 = torch.sigmoid(self.fc1(x))  # Sigmoid激活
        hidden_layer_2 = self.fc2(hidden_layer_1)    # 不是分类问题，不采用激活函数
        output_layer = self.svar(hidden_layer_2, h)  # 使用SVAR来计算输出
        return output_layer

    def svar(self, hidden_layer_2, h):
        global device
        X = h.T
        # X = X.cuda()
        n_features = X.shape[0]
        n_samples = X.shape[1]
        matrix_num = self.features * self.features
        M_iters = self.lags + 1
        result = torch.zeros((n_features, n_samples), device=device)
        for t in range(n_samples):
            if t - self.lags < 0:
                continue
            M_taus = torch.zeros((M_iters, self.features, self.features), device=device)
            for i in range(M_iters):
                a = i * matrix_num
                b = (i + 1) * matrix_num
                M_taus[i] = hidden_layer_2[t - self.lags, a:b].reshape(self.features, self.features)

            estimated = torch.zeros((n_features, 1), device=device)
            for tau in range(self.lags + 1):
                mid = X[:, t - tau].reshape(-1, 1).to(device)
                estimated += torch.matmul(M_taus[tau], mid)
            result[:, t] = estimated.reshape(-1)
        
        result = result[:, self.lags:].T
        # result = result.to(h.device)
        return result
    
def train(model, train_x, batch_size, epochs, lags, optimizer, criterion):
    
    model.train()
    train_loss_epoch = []
    batch = int((len(train_x) - lags) / batch_size)
    # model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(batch):
            start = lags + i * batch_size
            end = start + batch_size
            # inputs = torch.tensor(train_x[start:end], dtype=torch.float32) # 将数据移动到GPU
            # h = torch.tensor(train_x[start - lags:end], dtype=torch.float32)
            inputs = train_x[start:end]
            h = train_x[start - lags:end]
            
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs, h) # 前向传播

            loss = criterion(outputs.to(device), inputs)  # 使用MSE损失计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            total_loss += loss.item()
        
        avg_loss = total_loss / batch
        train_loss_epoch.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')
        
    return model, train_loss_epoch
    

 def testonline(model, test_x, lags):

    global mm
    
    model.eval()  # 设置为评估模式

    # inputs = torch.tensor(test_x[lags:,:], dtype=torch.float32)
    # h = torch.tensor(test_x, dtype=torch.float32)
    inputs = test_x[lags:,:]
    h = test_x
    with torch.no_grad():
        outputs = model(inputs, h)  # 不计算梯度
        outputs = outputs.cpu()  # 将结果移回CPU

    x = mm.inverse_transform(inputs.cpu())
    # x = x[lags:,:]
    y = mm.inverse_transform(outputs)

    col = inputs.shape[1]
    accuracy = np.zeros(col)
    r2 = np.zeros(col)
    for i in range(col):
        accuracy[i]=get_mape(x[-1,i], y[-1,i])
        r2[i]=get_r2(x[:,i], y[:,i])

    return accuracy, x, y, r2

def get_mape(x, y):
    """
    :param x:真实值
    :param y:预测值
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))

def get_r2(x,y):
    return r2_score(y_true=x, y_pred=y)

def get_adjustr2(x,y):
    r2=r2_score(y_true=x, y_pred=y)
    [n,p]=r2.shape
    adjust_r2=1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    return adjust_r2

def load_data(a, b):
    global device
    df = pd.read_csv('dataF106.csv', encoding='gbk')
    df = df.iloc[a:b, :]
    df.drop(['Motor_T2', 'MB_T2', 'Bearing_T3'], axis=1, inplace=True)
    df = df.to_numpy()
    
    mm = MinMaxScaler()
    df_t = mm.fit_transform(df)
    df_t = torch.tensor(df_t, dtype=torch.float32, device=device)
    train_x = df_t
    test_x = df_t
    train_y = df_t
    test_y = df_t
    return train_x, train_y ,test_x ,test_y, mm


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    batchsize = 10
    lags = 2
    features = 6
    learning_rate = 0.005
    epochs = 30

    a = 20000
    b = 20186  # b - train_size > lags

    train_size = 182 # batch = train_size - lags  /  batchsize
    train_loss=[]
    train_loss_epoch = []
    train_x, train_y, test_x, test_y, mm = load_data(a, b)
    # train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    # train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    # test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    # test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    model = BP(batchsize, lags, features, learning_rate).to(device) # 将模型移到GPU
    criterion = nn.MSELoss() # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_train = train_x.shape[0] - train_size
    number = batchsize + lags

    r2_list_1 = []
    r2_list_2 = []
    r2_list_3 = []
    r2_list_4 = []
    r2_list_5 = []
    r2_list_6 = []

    trainloss_list = []
    Running_time_list = []

    accuracy_list_1 = []
    accuracy_list_2 = []
    accuracy_list_3 = []
    accuracy_list_4 = []
    accuracy_list_5 = []
    accuracy_list_6 = []

    y_calc_1 = []
    y_calc_2 = []
    y_calc_3 = []
    y_calc_4 = []
    y_calc_5 = []
    y_calc_6 = []

    y_ori_1 = []
    y_ori_2 = []
    y_ori_3 = []
    y_ori_4 = []
    y_ori_5 = []
    y_ori_6 = []

    # create zero tensors with the correct shape and place them on the same device as the model/data
    M0 = torch.zeros((num_train - lags, features, features), device=device)
    M1 = torch.zeros((num_train - lags, features, features), device=device)
    M2 = torch.zeros((num_train - lags, features, features), device=device)

    for i in range(lags, num_train):
    # for i in range(1):
        
        train_loss_epoch = []
        righth = i + train_size
        train_data = train_x[i:righth,:]
        start = time.time()
        
        # model, train_loss_epoch = train(model, train_x=train_x_tensor, batch_size=batchsize, epochs=epochs, lags=lags, optimizer=optimizer, criterion=criterion)
        model, train_loss_epoch = train(model, train_x=train_data, batch_size=batchsize, epochs=epochs, lags=lags, optimizer=optimizer, criterion=criterion)
        end = time.time()
        Running_time_list.append(end - start)

        # test_x = train_data[:-number,:]
        test_x = train_data[number:,:]

        # call test_online with tensor on correct device
        accuracy, origin, calc, r2 = testonline(model, test_x, lags=lags)
        # accuracy, res, testy = test_online(test_data, lags=lags)
        # print(accuracy)

        # res_np = to_numpy(res)
        # testy_np = to_numpy(testy)
        # r2_np = to_numpy(r2)
        # z2_np = to_numpy(z2)

        r2_list_1.append(r2[0])
        r2_list_2.append(r2[1])
        r2_list_3.append(r2[2])
        r2_list_4.append(r2[3])
        r2_list_5.append(r2[4])
        r2_list_6.append(r2[5])
        # r2_list_7.append(r2[6])
        # r2_list_8.append(r2[7])
        # r2_list_9.append(r2[8])
        # accuracy_list.append(accuracy)
        # ensure accuracy elements are python scalars when possible

        accuracy_list_1.append(accuracy[0])
        accuracy_list_2.append(accuracy[1])
        accuracy_list_3.append(accuracy[2])
        accuracy_list_4.append(accuracy[3])
        accuracy_list_5.append(accuracy[4])
        accuracy_list_6.append(accuracy[5])

        trainloss = np.mean(train_loss_epoch)

        trainloss_list.append(trainloss)

        # use numpy versions for appending
        y_calc_1.append(calc[-1,0])
        y_ori_1.append(origin[-1:,0])
        y_calc_2.append(calc[-1,1])
        y_ori_2.append(origin[-1:,1])
        y_calc_3.append(calc[-1,2])
        y_ori_3.append(origin[-1:,2])
        y_calc_4.append(calc[-1,3])
        y_ori_4.append(origin[-1:,3])
        y_calc_5.append(calc[-1,4])
        y_ori_5.append(origin[-1:,4])
        y_calc_6.append(calc[-1,5])
        y_ori_6.append(origin[-1:,5])

    print('准确率：')
    print(np.mean(accuracy_list_1))
    print(np.mean(accuracy_list_2))
    print(np.mean(accuracy_list_3))
    print(np.mean(accuracy_list_4))
    print(np.mean(accuracy_list_5))
    print(np.mean(accuracy_list_6))
    # print(np.mean(accuracy_list_7))
    # print(np.mean(accuracy_list_8))
    # print(np.mean(accuracy_list_9))
    print('Running time: %s Seconds'%(np.mean(Running_time_list)))
    print('训练误差：')
    print(np.mean(trainloss_list))
    print('r2：')
    print(np.mean(r2_list_1))
    print(np.mean(r2_list_2))
    print(np.mean(r2_list_3))
    print(np.mean(r2_list_4))
    print(np.mean(r2_list_5))
    print(np.mean(r2_list_6))

~~~

	