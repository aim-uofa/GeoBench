<div align="center">
<img src="./assets/logo.png" width="128"/>

# ⚡ GeoBench: Benchmarking and Analyzing <br> Geometry Estimation Models

🔰 [Project Page](https://yongtaoge.github.io/projects/geobench/), 📑 [Paper](https://arxiv.org/abs/2406.12671)
    
[Yongtao Ge]()<sup>1,</sup><sup>2</sup>, [Guangkai Xu]()<sup>1</sup>, [Zhiyue Zhao]()<sup>1</sup>, [Libo Sun]()<sup>2</sup>, [Zheng Huang]()<sup>1</sup>, [Yanlong Sun]()<sup>3</sup>, [Hao Chen]()<sup>1</sup>, [Chunhua Shen]()<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[The University of Adelaide](https://www.adelaide.edu.au/aiml/), &nbsp;
<sup>3</sup>[Tsinghua University](https://www.tsinghua.edu.cn/en/) &nbsp;


</div>

> This toolbox streamlines the use and evaluation for state-of-the-art discriminative and generative geometry estimation models, including:

- [x] [Metric3D-V2](https://arxiv.org/abs/2404.15506)
- [x] [Depth-Anything-V2](https://arxiv.org/abs/2406.09414)
- [x] [Depth-Anything](https://arxiv.org/abs/2401.10891)
- [ ] [UniDepth](https://arxiv.org/abs/2403.18913)
- [x] [DSINE](https://arxiv.org/abs/2403.00712)
- [x] [Marigold](https://arxiv.org/abs/2312.02145)
- [x] [DMP](https://arxiv.org/abs/2311.18832)
- [x] [Genpercept](https://arxiv.org/abs/2403.06090)
- [x] [Geowizard](https://arxiv.org/abs/2403.12013)
- [x] [DepthFM](https://arxiv.org/abs/2403.13788)

We would also support multi-view geometry models in near future:
- [ ] [DUSt3R](https://arxiv.org/abs/2312.14132)
- [ ] [MASt3R](https://arxiv.org/abs/2406.09756)

## Install
```
pip install -r requirements.txt
pip install -e . -v
```

## Inference Demos
```
# inference Marigold
sh scripts/run_marigold.sh

# inference Metric3D
sh scripts/run_metric3d.sh

# inference Depth-Anything
sh scripts/run_depthanything.sh

# inference GenPercept
sh scripts/run_genpercept.sh

# inference DSINE
sh scripts/run_dsine.sh
```

## Benchmarks and Model Zoo

Stay tuned, comming soon.


## Citation
If you find the toolbox useful for your project, please cite our paper:
```
@article{ge2024geobench,
    title={GeoBench: Benchmarking and Analyzing Monocular Geometry Estimation Models},
    author={Ge, Yongtao and Xu, Guangkai, and Zhao, Zhiyue and Huang, zheng and Sun, libo and Sun, Yanlong and Chen, Hao and Shen, Chunhua},
    journal={arXiv preprint arXiv:2406.12671},
    year={2024}
}

```
## License

GeoBench is released under the [Apache 2.0 license](LICENSE).