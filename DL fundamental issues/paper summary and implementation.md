BN

以`mmcv`的实现为例。一如既往，注册组件NORM_LAYERS

```python
from .registry import NORM_LAYERS

NORM_LAYERS.register_module('BN', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', module=nn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', module=nn.BatchNorm3d)
NORM_LAYERS.register_module('SyncBN', module=SyncBatchNorm)
NORM_LAYERS.register_module('GN', module=nn.GroupNorm)
NORM_LAYERS.register_module('LN', module=nn.LayerNorm)
NORM_LAYERS.register_module('IN', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', module=nn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', module=nn.InstanceNorm3d)
```

NORM_LAYERS注册类的build_from_cfg控制逻辑，对常见的BN2d来说，核心语句就是

layer = norm_layer(num_features, **cfg_)

其中norm_layer为nn.BatchNorm2d。

```python
def build_norm_layer(cfg, num_features, postfix=''):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
```



`ResNet`系列

要点介绍

`ResNet-34`：

![img](https://robotics-robotics.feishu.cn/space/api/box/stream/download/asynccode/?code=MTc1ODUzYTA3OGRlYzZlNTg0YjNiNTQzZDNhOTMxMTJfVk5DQ3NzN0ZhZzByRGxHREVpYzFXU3BJVmNoMEdETHFfVG9rZW46Ym94Y25MOFIyQUtWRFZTeURrZlN2VGVldW9nXzE2NDg0NTU1Mjk6MTY0ODQ1OTEyOV9WNA)

![img](https://robotics-robotics.feishu.cn/space/api/box/stream/download/asynccode/?code=MjVkNTZlZDhkZTc0ZTkwMDMwZjA2NmQ2OWE5MTRiOThfQ29NemlnYzlGRGw4V1hGUXo0MmtmYTFsV01qTW9WelNfVG9rZW46Ym94Y25mRzR5dDZpTXZud1NRV1Z1Yk9sWU9kXzE2NDg0NTU1Mjk6MTY0ODQ1OTEyOV9WNA)

网络的`stem`是一个7x7，stride=2，padding=3的卷积，输出通道数是64，再通过`BN`与`ReLU`。

接下来（默认）通过4个`stage`（上图中同一个`stage`的操作有同一种底色），每个`stage`由若干个残差结构`block`构成，除了第一个`stage`外，每个`stage`的第一个`block`会进行宽高减半同时通道数翻倍的操作（上图中这些`block`的直通分支用虚线表示其需要进行相应的下采样操作）。`ResNet-34`由4个`stage`组成，分别由3、4、6、3个`block`组成。

`Block`有两种，较浅的模型`ResNet-18`与`ResNet-34`使用`BasicBlock`，`ResNet-50`，`ResNet-101`，`ResNet-152`使用`Bottleneck`。代码中分别实现了这两种`Block`。在`ResNet`模型实现中首先通过`ResNet`的层数确定使用何种`Block`以及每个`stage`中`block`的数量，再依次构建`block`与`stage`，其中每一个`stage`的第一个`block`单独处理下采样的问题。



![img](https://robotics-robotics.feishu.cn/space/api/box/stream/download/asynccode/?code=YTQzNDFmZjhlYTFkMmZhYzYxMDgwYmJmZDVmMzI2NzlfeHVTY3JSRGhRU21XTFJZRWZ0SjVsc0F5cFNtYVlmRExfVG9rZW46Ym94Y25OWmVvUHdPU3Y2MXhydXNYWmJFRmhiXzE2NDg0NTU1Mjk6MTY0ODQ1OTEyOV9WNA)

![img](https://robotics-robotics.feishu.cn/space/api/box/stream/download/asynccode/?code=YjViYzgzNmQ1YzU5YTgwZThmMTIwMjc1MWQ4ODliNmFfWjdpZDl3cmdpalJlbXVHMmkxOEZPUTZTREFkY0RLMmdfVG9rZW46Ym94Y25yRVVsa0Z0SG1mVEQzUzV5R3JnYW9nXzE2NDg0NTU1Mjk6MTY0ODQ1OTEyOV9WNA)



`ResNet v2`论文对比了不同的`BN`与`ReLU`的位置带来的影响。注意，代码中实现的是最原始的结构，即

![img](https://robotics-robotics.feishu.cn/space/api/box/stream/download/asynccode/?code=NzBiMjE4OWUwZTJkOGQ2NzliYjg3NTEwMTk1NDlhNmNfaDg2bWNqbFNwczdicVNnRW5kN0tDZkFCM0VqSnpqN3VfVG9rZW46Ym94Y250c0l2NG1Fcm1vbEVXa0VHVlJEaGt4XzE2NDg0NTU1Mjk6MTY0ODQ1OTEyOV9WNA)

以`mmclassification`的实现为例

首先定义了两种`Block`

`BasicBlock`

```python
class BasicBlock(BaseModule):  # 直筒等尺寸
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1  # 对于BasicBlock，expansion等于1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
    
    def forward(self, x):
        def _inner_forward(x):
            identity = x
			# conv1 -> BN1 -> ReLU -> conv2 -> BN2
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
			# if the current block is 1st in the concerned stage other than 1st stage, downsample needed.
            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
		# final ReLU
        out = self.relu(out)

        return out
```



`Bottleneck`

```python
class Bottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
			# conv1 -> BN1 -> ReLU -> conv2 -> BN2 -> ReLU -> conv3 -> BN3
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
		# final ReLU
        out = self.relu(out)

        return out
```



基于`Block`类定义`Stage`

```python
class ResLayer(nn.Sequential):
    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        # 1 for BasicBlock, 4 for Bottleneck
        self.expansion = get_expansion(block, expansion)  

        downsample = None
        # deal with 1st block in each stage to downsample (define the conv, optionally the avg pooling)
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            # if used avg pooling to downsample
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            # used avg pooling conv_stride = 1, otherwise conv_stride should be 2
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion, # 1 for BasicBlock, 4 for Bottleneck
                stride=stride,
                downsample=downsample, # difined above
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels # downsample done
        # other blocks in the current stage, downsample = None
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)
```



`ResNet`类首先定义好了各个`ResNet`使用的`Block`类型与每个`Stage`的`Block`数量。



