# 多模态对抗与孪生自增强推荐方法


<h2>Multi-modal Adversarial and Siamese Self-augentation Recommended method</h2>
模型结构示意图
<p align="center">
<img src="./MASSR.png" alt="MASSR" />
</p>


<h2>数据集参考</h2>

分别来自于“Amazon”、“Tiktok”、“Allrecipes”平台的公开历史数据集。
经预处理后的部分数据可参考 MASSR-data：
https://pan.baidu.com/s/1AMHbYyvh4IlHnCKHir6vBg 提取码: 6666


<h2>环境参考</h2>

* Python：3.9.13
* Pytorch：1.13.0+cu116
* dgl-cuda11.6：0.9.1post1

<h2>运行参考</h2>

启动实验及数据集选择示例：
‘’‘
cd MMSSL
python ./main.py --dataset {DATASET}
’‘’
