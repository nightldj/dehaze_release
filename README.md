# dehaze_release

This is the PyTorch code for [''Strong Baseline for Single Image Dehazing with Deep Features and Instance Normalization''](http://bmvc2018.org/contents/papers/0821.pdf) publisehd in BMVC 2018.  The arxiv version is [here](https://arxiv.org/abs/1805.03305). 

The pre-trained model can be found [here](https://drive.google.com/file/d/1D54n3ODhap3L-ytJ0mNI_G9q6qOJrQx0/view?usp=sharing). 

To test the pre-trained model, put the downloaded models in folder named ''models'', put the [RESIDE standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) in ''data'', and run 
```
python main.py --trans-flag in --use-bn in  --test-flag --test-batch-size 8 --gpuid 0 --load-model models/dehaze_release.pth --save-image output
```

The dehazed images can be found in folder ''output''. The pre-trained model could achieve PSNR 27.79 and SSIM 0.9556 on RESIDE_standard, evaluated by the matlab script provided on the dataset webpage. 

To train a model, please run
```
python main.py --trans-flag in --use-bn in --batch-size 16 --test-batch-size 8 --optm sgd --lr 0.1 --lr-freq 30 --epochs 60 --rec-w 1 --per-w 1  --print-freq 200 --gpuid 0,1,2,3
```

## citation 
@article{xu2018effectiveness,
  title={Strong Baseline for Single Image Dehazing with Deep Features and Instance Normalization},
  author={Xu, Zheng and Yang, Xitong and Li, Xue and Sun, Xiaoshuai},
  journal={BMVC},
  year={2018}
}


### acknowledgement
We thank the released PyTorch code and model of [WCT style transfer](https://github.com/sunshineatnoon/PytorchWCT). 
