# CBDNet
This is a official implementation for our paper: "[Oil spill contextual and boundary-supervised detection network based on marine SAR images](https://ieeexplore.ieee.org/abstract/document/9568691)" has been published on <font size=5>**IEEE Transactions on Geoscience and Remote Sensing**</font> by Qiqi Zhu , Yanan Zhang , Ziqi Li , Xiaorui Yan , Qingfeng Guan , Yanfei Zhong , Liangpei Zhang and Deren Li.

## Dataset Download Link
Link: https://github.com/CUG-URS/Deep-SAR-Oil-Spill-Dataset

## Usage
### Train
**train.py** 
run: python train.py 
```
1.Dataset path (palsar, Sentinel)
ROOT = '.datasets/train/palsar/'
2.CBDNet is the network, dice_bce_loss is the loss function, and the learning rate is 2e-4
solver = MyFrame(CBDNet, dice_bce_loss, 2e-4) 
3.Name of weights document
NAME = 'palsar_CBDNet' 
```

### Test
**test.py**
run: python test.py
```
1.Dataset path
source = '.datasets/test/palsar/sat/'
2.Loading Network
solver = TTAFrame(CBDNet)
3.Loading the weights file
solver.load('.weights/palsar_CBDNet.th')
4.Test Results Documentation
target = '.submits/palsar_CBDNet/'
```


### Precision evaluation
**t1p3-Iou.py**  
run: python t1p3-Iou.py
```
1.truth label
name_truth = '.datasets/test/palsar/gt/'
2.Predicted results
name_pred = '.submits/palsar_CBDNet/'

```

## Citation
If you find our work useful for your research, please consider citing our paper:  
```
@article{zhu2021oil,
  title={Oil spill contextual and boundary-supervised detection network based on marine SAR images},
  author={Zhu, Qiqi and Zhang, Yanan and Li, Ziqi and Yan, Xiaorui and Guan, Qingfeng and Zhong, Yanfei and Zhang, Liangpei and Li, Deren},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--10},
  year={2021},
  publisher={IEEE}
} 
```

## Contact us 
```
You can contact the e-mail zhuqq@cug.edu.cn if you have further questions about the usage of codes and datasets.
For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).
The homepage of our academic group is: http://grzy.cug.edu.cn/zhuqiqi/en/index.htm.
Date: Dec 4, 2023
``` 
