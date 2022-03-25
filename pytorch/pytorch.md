# pytorch保存模型与加载模型

https://pytorch.org/tutorials/beginner/saving_loading_models.html#

## state_dict

pytorch中有两类对象拥有状态字典，一种是model（nn.module），一种是optimizer。前者的状态字典就是一个python字典，把model的各个layer（如nn.module中自定义的属性，举例self.conv1）映射到参数张量。后者则是记录优化器的使用超参。

## 保存/加载state_dict用于推理（官方推荐）

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

## 保存/加载完整模型

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

