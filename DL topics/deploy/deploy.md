pytorch->onnx

pytorch训练后生成的通常为pth模型（或者说，载入网络结构与参数后是一个torch.nn.Module对象），可以使用pytorch自带的API转换为onnx模型。

```python
def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
           input_names=None, output_names=None, operator_export_type=None,
           opset_version=None, _retain_param_name=True, do_constant_folding=True,
           example_outputs=None, strip_doc_string=True, dynamic_axes=None,
           keep_initializers_as_inputs=None, custom_opsets=None, enable_onnx_checker=True,
           use_external_data_format=False)
```

model (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction)，如果是torch.nn.Module，方法内部会转为TorchScript graph。

args (tuple or torch.Tensor)，含义是model的输入参数，支持3种给定形式。对于图像问题，通常给定一个batchsize的图像张量即可。