---
backbone:
- convNext-Tiny
integrating: True
domain:
- cv
frameworks:
- pytorch
language:
- en
- ch
license: Apache License 2.0
metrics:
- Line Accuracy
finetune-support: True
tags:
- OCR
- Alibaba
- 文字识别
tasks:
- ocr-recognition

studios:
- damo/cv_ocr-text-spotting

datasets:
  test:
  - damo/ICDAR13_HCTR_Dataset

widgets:
  - task: ocr-recognition
    inputs:
      - type: image
    examples:
      - name: 1
        inputs:
          - name: image
            data: http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg        
---


# 读光文字识别
## News
- 2023年6月：
    - 新增轻量化端侧识别[LightweightEdge-通用场景](https://www.modelscope.cn/models/damo/cv_LightweightEdge_ocr-recognitoin-general_damo/summary)模型和轻量化端侧[行检测模型](https://www.modelscope.cn/models/damo/cv_proxylessnas_ocr-detection-db-line-level_damo/summary)。
- 2023年4月：
    - 新增训练/微调时读取本地数据集的lmdb，用训练/微调后的模型继续识别，详见代码示例。
- 2023年3月：
    - 新增训练/微调流程，支持自定义参数及数据集，详见代码示例。
- 2023年2月：
    - 新增业界主流[CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)模型。

## 传送门
各场景文本识别模型：
- [ConvNextViT-通用场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo/summary)
- [ConvNextViT-文档印刷场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary)
- [ConvNextViT-自然场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-scene_damo/summary)
- [ConvNextViT-车牌场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary)
- [CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)

各场景文本检测模型：
- [SegLink++-通用场景行检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-line-level_damo/summary)
- [SegLink++-通用场景单词检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-word-level_damo/summary)
- [DBNet-通用场景行检测](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)

整图OCR能力：
- [整图OCR-多场景](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)

欢迎使用！

## 模型描述
- 文字识别，即给定一张文本图片，识别出图中所含文字并输出对应字符串。
- 本模型主要包括三个主要部分，Convolutional Backbone提取图像视觉特征，ConvTransformer Blocks用于对视觉特征进行上下文建模，最后连接CTC loss进行识别解码以及网络梯度优化。识别模型结构如下图：   

<p align="center">
    <img src="./resources/ConvTransformer-Pipeline.jpg"/> 
</p>

## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字内容，具体地，模型输出内容以字符串形式输出。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。
- 注：输入图片应为包含文字的单行文本图片。其它如多行文本图片、非文本图片等可能没有返回结果，此时表示模型的识别结果为空。

## 模型推理
在安装完成ModelScope之后即可使用ocr-recognition的能力。(在notebook的CPU环境或GPU环境均可使用)
- 使用图像的url，或准备图像文件上传至notebook（可拖拽）。
- 输入下列代码。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo')

### 使用url
img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg'
result = ocr_recognition(img_url)
print(result)

### 使用图像文件
### 请准备好名为'ocr_recognition.jpg'的图像文件
# img_path = 'ocr_recognition.jpg'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)
# print(result)
```

### 模型可视化效果
以下为模型的可视化文字识别效果。

<p align="center">
    <img src="./resources/rec_result_visu.png" width="400" /> 
</p>

### 模型局限性以及可能的偏差
- 模型是在中英文手写数据集上训练的，在其他语言或其他场景的数据上有可能产生一定偏差，请用户自行评测后决定如何使用。
- 当前版本在python3.7的CPU环境和单GPU环境测试通过，其他环境下可用性待测试。

## 模型微调/训练
### 训练数据及流程介绍
- 本文字识别模型训练数据集来自收集数据，训练数据数量约2M。
- 本模型参数随机初始化，然后在训练数据集上进行训练，在32x300尺度下训练20个epoch。

### 模型微调/训练示例
#### 训练数据集准备
示例采用[ICDAR13手写数据集](https://modelscope.cn/datasets/damo/ICDAR13_HCTR_Dataset/summary)，已制作成lmdb，数据格式如下
```
'num-samples': number,
'image-000000001': imagedata,
'label-000000001': string,
...
```
详情可下载解析了解。

#### 配置训练参数并进行微调/训练
参考代码及详细说明如下
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile, DownloadMode

### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中 
### 当前notebook中版本为1.3.2，请手动更新，建议使用GPU环境

model_id = 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo' 
cache_path = snapshot_download(model_id) # 模型下载保存目录
config_path = os.path.join(cache_path, ModelFile.CONFIGURATION) # 模型参数配置文件，支持自定义
cfg = Config.from_file(config_path)

# 构建数据集，支持自定义
train_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset', 
    split='test',
    namespace='damo',
    test_mode=False)

train_dataset = MsDataset.load( 
    dataset_name=train_data_cfg.name,
    split=train_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

test_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset',
    split='test',
    namespace='damo',
    test_mode=True)

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

tmp_dir = tempfile.TemporaryDirectory().name # 模型文件和log保存位置，默认为"work_dir/"

# 自定义参数，例如这里将max_epochs设置为15，所有参数请参考configuration.json
def _cfg_modify_fn(cfg):
    cfg.train.max_epochs = 15
    return cfg

####################################################################################

'''
使用本地文件
    lmdb: 
        构建包含下列信息的lmdb文件 (key: value)
        'num-samples': 总样本数,
        'image-000000001': 图像的二进制编码,
        'label-000000001': 标签序列的二进制编码,
        ...
        image和label后的index为9位并从1开始
下面为示例 (local_lmdb为本地的lmdb文件)
'''

# train_dataset = MsDataset.load( 
#     dataset_name=train_data_cfg.name,
#     split=train_data_cfg.split,
#     namespace=train_data_cfg.namespace,
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
#     local_lmdb='./local_lmdb')

# test_dataset = MsDataset.load(
#     dataset_name=test_data_cfg.name,
#     split=test_data_cfg.split,
#     namespace=train_data_cfg.namespace,
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
#     local_lmdb='./local_lmdb')

####################################################################################

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    work_dir=tmp_dir,
    cfg_modify_fn=_cfg_modify_fn)

# 模型训练
trainer = build_trainer(name=Trainers.ocr_recognition, default_args=kwargs)
trainer.train()
```

#### 用训练/微调后的模型进行识别
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

ep_num = 3  # 选择模型checkpoint
cmd = 'cp {} {}'.format('./work_dir/epoch_%d.pth' % ep_num, './work_dir/output/pytorch_model.pt')  # 'work_dir'为configuration中设置的路径，'output'为输出默认路径
os.system(cmd)
ocr_recognition = pipeline(Tasks.ocr_recognition, model='./work_dir/output' )
result = ocr_recognition('http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_icdar13.jpg')
print(result)
```

#### ONNX模型使用
```python
import torch
import numpy as np
import torch.nn.functional as F
import onnxruntime as rt
import cv2

def keepratio_resize(img):
    cur_ratio = img.shape[1] / float(img.shape[0])
    mask_height = 32
    mask_width = 804
    if cur_ratio > float(mask_width) / mask_height:
        cur_target_height = mask_height
        cur_target_width = mask_width
    else:
        cur_target_height = mask_height
        cur_target_width = int(mask_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
    mask[:img.shape[0], :img.shape[1], :] = img
    img = mask
    return img

img = cv2.imread('ocr_recognition.jpg') # 请在替换本地测试图片路径
img = keepratio_resize(img)
img = torch.FloatTensor(img)
chunk_img = []
for i in range(3):
    left = (300 - 48) * i
    chunk_img.append(img[:, left:left + 300, :])
merge_img = torch.cat(chunk_img, 0)
data = merge_img.view(3, 32, 300, 3) / 255.
data = data.permute(0, 3, 1, 2).cuda() 
input_data = data.cpu().numpy()

# inference
sess = rt.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
output_name= sess.get_outputs()[0].name
res = sess.run([output_name], {input_name: input_data})
outprobs = F.softmax(torch.tensor(res[0]), dim=-1)
preds = torch.argmax(outprobs, -1)

# load dict and CTC decode
# vocab.txt可从模型主页下载
labelMapping = dict()
with open('vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    cnt = 2
    for line in lines:
        line = line.strip('\n')
        labelMapping[cnt] = line
        cnt += 1

batchSize, length = preds.shape
final_str_list = []
for i in range(batchSize):
    pred_idx = preds[i].cpu().data.tolist()
    last_p = 0
    str_pred = []
    for p in pred_idx:
        if p != last_p and p != 0:
            str_pred.append(labelMapping[p])
        last_p = p
    final_str = ''.join(str_pred)
    final_str_list.append(final_str)

print(final_str_list)
```