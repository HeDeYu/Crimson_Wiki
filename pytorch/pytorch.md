# `Tensor`类

与`numpy`的`ndarray`类似，通常在底层共享内存，区别是tensor能在GPU或者其它加速硬件上运算，同时对自动微分做了优化。

## `Tensor`实例化

基于数据（嵌套list对象）直接实例化（构造函数）

通过numpy的array实例化（类方法from_numpy）

通过已有的Tensor对象实例化（复制构造函数）

指定尺寸的随机/常数实例化（各种类方法）

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

`tensor`转为`numpy` `array`

```python
t = torch.ones(5)
print(f"t: {t}")  # t: tensor([1., 1., 1., 1., 1.])
n = t.numpy()
print(f"n: {n}")  # n: [1. 1. 1. 1. 1.]
```

注意，cpu上tensor与numpy array数据是同一片内存，因此对其中一个对象的修改会反映到另一个对象。



## `Tensor`类属性

3个属性shape，dtype，device



## `Tensor`类方法

### 从`cpu`复制数据到`gpu`

```python
tensor = tensor.to("cuda")
```

### `numpy`标准索引/切片/赋值

与`numpy`操作一致

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

### 张量拼接

API torch.cat，dim形参指定拼接维度

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
```

API torch.stack

### 矩阵乘法

3种方式（对象方法，类方法，运算符重载）

```python
a = torch.rand((2, 3,))
b = torch.rand((3, 2,))
c1 = a @ b
c2 = a.matmul(b)
c3 = torch.rand((2, 2,))
torch.malmul(a, b, out=c3)
```

### 元素级（逐元素/element-wise）乘法

与矩阵乘法类似的3种方式

```python
a = torch.rand((2, 3,))
b = torch.rand((2, 3,))
d1 = a * b
d2 = a.mul(b)
d3 = torch.rand((2, 3,))
torch.mul(a, b, out=d3)
```

### 单元素张量转数值

API item()

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))  # 12.0 <class 'float'>
```

### 原地操作

通常API带有下划线后缀的，为原地操作，通常不建议原地操作，以免计算中间loss时出错。



### `Tensor`的`view()`

https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

张量的`view`方法返回与原张量共享内存的新张量，不进行数据复制的情况下改变张量的维度表达，或者以新的数据类型解释数据。



更多操作查阅

https://pytorch.org/docs/stable/torch.html



# `Datasets` & `DataLoaders`

总结到另一篇文档 torch_utils_data.md。



# `Build and use models`

注意，通常不要直接使用model的forward，因为实际训练时，调用model本身时除了forward还有autograd的一些动作。



# Optimization



# Autograd

## requires_grad

## grad_fn

## backward()与retain_graph

## 禁用autograd

应用场景

### 上下文torch.no_grad()

### 张量方法detach()

## DAG



# pytorch模型保存、加载、部署运行



https://pytorch.org/tutorials/beginner/saving_loading_models.html#

https://pytorch.org/docs/stable/jit.html

https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

https://pytorch.org/tutorials/advanced/cpp_export.html

本篇基于上述链接，基本就是翻译，只是按照自己的理解调整了一下段落位置与从属关系。



## 保存/加载model的state_dict用于推理

### state_dict

pytorch中有两类对象拥有状态字典，一种是model（nn.module），一种是optimizer。前者的状态字典就是一个python字典，把model的各个layer（如nn.module中自定义的属性，举例self.conv1）映射到参数张量。后者则是记录优化器的使用超参。

### 保存/加载state_dict

保存模型状态字典

```python
torch.save(model.state_dict(), PATH)      # 传统上保存pt或pth文件
```

加载模型状态字典

```python
model = TheModelClass(*args, **kwargs)    # 先生成模型architecture
model.load_state_dict(torch.load(PATH))   # 载入参数，注意先反序列化再调用load_state_dict()
model.eval()                              # eval确保dropout与BN等设置正确
```

可以看到，状态字典本身并没有包含模型architecture，需要先初始化model。

注意，model.state_dict()返回的是参数的引用，而非复本，所以使用本方式动态保存训练过程中的模型参数时，需要对模型状态字典使用序列化或者深复制。



### 部分加载state_dict

加载model的state_dict时，如果保存的model与加载目标的layers不完全匹配，譬如保存的模型有一些layers是加载目标模型没有的，或者反过来，加载目标模型有些layers是保存的模型没有的（这很常见，譬如只导入主干网络的预训练权重，但是下游任务相关的参数没有/不使用预训练权重），可以设置load_state_dict()的strict参数为false。

保存模型A的权重

```python
torch.save(modelA.state_dict(), PATH)
```

加载到模型B，模型A与模型B的layers不需要完全匹配。

```python
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

另外，如果模型A与模型B某一个layer仅仅名字不一样（在类的定义中的名字），但又想导入的时候，可以修改model state dict中该layer的key强制匹配。



## 跨设备加载model的state_dict用于推理

### GPU保存，CPU加载

GPU保存

```python
torch.save(model.state_dict(), PATH)
```

CPU加载，在torch.load()调用时使用map_location参数。

```python
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```



### GPU保存，GPU加载

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```



### CPU保存，GPU加载

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```



### 保存数据并行模型

使用数据并行多卡训练时，使用以下API进行模型的state dict保存。

```python
torch.save(model.module.state_dict(), PATH)
```





## 保存/加载完整模型用于推理

保存完整模型

```python
torch.save(model, PATH)
```

加载完整模型

```python
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```

可以看到，完整模型包含了模型architecture本身，不需要先初始化model，但是需要有模型类的定义。



## 保存/加载checkpoint用于推理/恢复训练

### 保存/加载单个模型及其训练状态

checkpoint除了包括模型，还包括优化器信息、训练中止时的其它参数等。通常保存为.tar文件。

保存checkpoint，显式调用API保存模型的state_dict，优化器的state_dict以及其它训练状态参数。

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

加载checkpoint，显式设置模型的state_dict，优化器的state_dict以及其它训练状态参数。

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```



### 保存/加载多个模型及其训练状态

与保存/加载单个模型到checkpoint类似。通常保存为.tar文件。

```python
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)
```



```python
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```



torch.save()是一个序列化的过程，而序列化的内容是非常自由的。项目中涉及多个模型的场景也是普遍的，譬如GAN，sequence-to-sequence，model ensemble。



## TorchScript保存/加载

### TorchScript

https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

可以把TorchScript理解为一个jit编译器，把python源码编译为一个静态计算图。这样做的好处有（按官方）：

1. TorchScript代码可以被受限的python解释器唤醒，可以绕开（掩面，臭名昭著的）GIL
2. 可以移植到其它设备其它语言环境中使用
3. 编译优化，高效执行
4. 允许其它终端接口

### trace编译器与script编译器

有两种方式把python源码module转为TorchScript对象，第一种是给module输入一个数据，追踪计算过程，生成静态计算图，这时候使用的是trace编译器，第二种是使用script编译器，直接把源码转为静态计算图。

生成的对象可以使用其graph和code属性来查看计算图和逻辑。

使用trace编译器的话对if-statement与loop-statement不会“起效”。

譬如对如下module的定义，带有控制流程。

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x
```

如果使用torch.jit.trace生成静态计算图后，查看其code属性时会看到并没有我们想要的逻辑，因为trace编译器是根据追踪运算流得到静态计算图，是基于实际发生的运算，所以只会记录其中一条分支。（以下的code属性有点问题，感觉应该是tensor -> tensor的函数才对）

```python
def forward(self,
    argument_1: Tensor) -> NoneType:
  return None
```

使用torch.jit.script生成静态计算图则会保留源码逻辑

```python
def forward(self,
    x: Tensor) -> Tensor:
  if bool(torch.gt(torch.sum(x), 0)):
    _0 = x
  else:
    _0 = torch.neg(x)
  return _0
```



### trace与script混合使用（todo）



### TorchScript保存与加载

保存TorchScript模型

```python
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
```

加载TorchScript模型，不需要有模型类的定义（因为已经是静态运算图了）。

```python
model = torch.jit.load('model_scripted.pt')
model.eval()
```



### TorchScript在C++环境下的部署（todo）

