# Entity-Aware and Motion-Aware Transformers for Language-driven Action

This is implementation for the paper "Entity-Aware and Motion-Aware Transformers for Language-driven Action" (**IJCAI 2022**)

```shell
# preparing environment
bash conda.sh
```

## Dataset Preparation
We use [VSLNet's](https://github.com/IsaacChanghau/VSLNet) data. The visual features can be download [here](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw), for CharadesSTA we use the "new" fold, and for TACoS we use the "old" fold, annotation and other details can be found [here](https://github.com/IsaacChanghau/VSLNet/tree/master/prepare)
and then modify the line 81~91 of "dataset/BaseDataset.py" to your own path.

## Quick Start
**Train**
```shell script
python main.py --cfg experiments/charades/EAMAT.yaml --mode train
python main.py --cfg experiments/tacos/EAMAT.yaml --mode train
```
a new fold "results" are created.

## Citation
If you feel this project helpful to your research, please cite our work.
```
@inproceedings{DBLP:conf/ijcai/YangW22,
  author    = {Shuo Yang and
               Xinxiao Wu},
  title     = {Entity-aware and Motion-aware Transformers for Language-driven Action
               Localization},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  pages     = {1552--1558},
  publisher = {ijcai.org},
  year      = {2022},
  url       = {https://doi.org/10.24963/ijcai.2022/216},
  doi       = {10.24963/ijcai.2022/216},
  timestamp = {Wed, 27 Jul 2022 16:43:00 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/YangW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
