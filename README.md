# README



## 预训练权重

需从以下两个网址下载权重pytorch_model.bin，一个放在./models/backbones/ERNIE中，一个放在./models/backbones/GPT中。

https://huggingface.co/nghuyong/ernie-health-zh/tree/main

https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main



## 文件位置

直接在./constants.py文件中修改即可。



## 运行

```
cd ./models/mgca/
python --gpus 1 --strategy ddp --precision 16 --img_encoder vit_base mgca_module.py
```



## 测试

./models/textgen中有个text_gen.py的生成代码，用的是beam search，可以生成给定prompt是输出的文本，不过这块接口写的很乱，建议重写。mgca_module.py中的主函数已经写好了encode和decode的过程，调用可得到词汇概率。