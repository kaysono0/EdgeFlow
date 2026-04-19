# MobileNetV2 量化对比报告

## 测试环境
- 设备: Mac M1 Pro
- 模型: MobileNetV2
- 运行时: ONNX Runtime
- 测试轮数: 100

## 正式对比结果

| 模型            |   中位数 (ms) |   P95 (ms) |   均值 (ms) |   模型大小 (MB) |
|:--------------|-----------:|-----------:|----------:|------------:|
| FP32          |      11.65 |      12.72 |     11.63 |       13.99 |
| FP16          |      29.84 |      31.26 |     29.89 |        7.03 |
| INT8 (静态 PTQ) |       2.1  |       2.58 |      2.27 |        3.91 |

## 本地图片集精度汇总

评估清单: `eval_manifest.csv`

| 模型            |   样本数 | Top-1 准确率   | 与 FP32 Top-1 一致率   |   平均 logits cosine |
|:--------------|------:|:------------|:-------------------|-------------------:|
| FP32          |     7 | 71.43%      | 100.00%            |           1        |
| FP16          |     7 | 71.43%      | 100.00%            |           1.00006  |
| INT8 (静态 PTQ) |     7 | 71.43%      | 100.00%            |           0.981145 |

## 样本级结果

| 图片             | 模型            | 标签         | Top-1 预测         | Top-1 概率   | 预测正确   | 与 FP32 Top-1 一致   |   logits cosine |
|:---------------|:--------------|:-----------|:-----------------|:-----------|:-------|:------------------|----------------:|
| test_dog.jpg   | FP32          | Samoyed    | Samoyed          | 83.03%     | 是      | 是                 |        1        |
| test_dog.jpg   | FP16          | Samoyed    | Samoyed          | 82.71%     | 是      | 是                 |        1.00008  |
| test_dog.jpg   | INT8 (静态 PTQ) | Samoyed    | Samoyed          | 80.64%     | 是      | 是                 |        0.981871 |
| dog2.jpg       | FP32          | Samoyed    | golden retriever | 48.11%     | 否      | 是                 |        1        |
| dog2.jpg       | FP16          | Samoyed    | golden retriever | 48.58%     | 否      | 是                 |        1.00038  |
| dog2.jpg       | INT8 (静态 PTQ) | Samoyed    | golden retriever | 54.29%     | 否      | 是                 |        0.981165 |
| tabby_cat.jpg  | FP32          | tabby      | tabby            | 67.04%     | 是      | 是                 |        1        |
| tabby_cat.jpg  | FP16          | tabby      | tabby            | 66.99%     | 是      | 是                 |        1.00004  |
| tabby_cat.jpg  | INT8 (静态 PTQ) | tabby      | tabby            | 66.19%     | 是      | 是                 |        0.99029  |
| tiger_cat.jpg  | FP32          | tiger cat  | jaguar           | 48.36%     | 否      | 是                 |        1        |
| tiger_cat.jpg  | FP16          | tiger cat  | jaguar           | 48.12%     | 否      | 是                 |        1.00009  |
| tiger_cat.jpg  | INT8 (静态 PTQ) | tiger cat  | jaguar           | 39.56%     | 否      | 是                 |        0.970115 |
| goldfish.jpg   | FP32          | goldfish   | goldfish         | 100.00%    | 是      | 是                 |        1        |
| goldfish.jpg   | FP16          | goldfish   | goldfish         | 100.00%    | 是      | 是                 |        0.999698 |
| goldfish.jpg   | INT8 (静态 PTQ) | goldfish   | goldfish         | 100.00%    | 是      | 是                 |        0.989643 |
| macaw.jpg      | FP32          | macaw      | macaw            | 99.98%     | 是      | 是                 |        1        |
| macaw.jpg      | FP16          | macaw      | macaw            | 100.00%    | 是      | 是                 |        0.999934 |
| macaw.jpg      | INT8 (静态 PTQ) | macaw      | macaw            | 99.96%     | 是      | 是                 |        0.981801 |
| sports_car.jpg | FP32          | sports car | sports car       | 73.17%     | 是      | 是                 |        1        |
| sports_car.jpg | FP16          | sports car | sports car       | 72.56%     | 是      | 是                 |        1.00018  |
| sports_car.jpg | INT8 (静态 PTQ) | sports car | sports car       | 72.40%     | 是      | 是                 |        0.973133 |

## 备注

- eval_manifest.csv:9 图片不存在: /Users/junsen/workspace/EdgeFlow/mobilenetv2-demo/models/mountain_bike.jpg
- INT8 (动态) 在 test_dog.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 dog2.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 tabby_cat.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 tiger_cat.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 goldfish.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 macaw.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
- INT8 (动态) 在 sports_car.jpg 上推理失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'

## 静态 PTQ 备注

- eval_manifest.csv:9 图片不存在: /Users/junsen/workspace/EdgeFlow/mobilenetv2-demo/models/mountain_bike.jpg

## 兼容性验证项

- INT8 (动态) 模型已生成: mobilenetv2_int8_dynamic.onnx (3.70 MB)
- INT8 (动态) 在当前后端不可执行: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/features/features.0/features.0.0/Conv_quant'
