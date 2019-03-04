# DeepMatchVO
Implementation of ICRA 2019 paper: [Beyond Photometric Loss for Self-Supervised Ego-Motion Estimation](https://arxiv.org/abs/1902.09103)
```
@inproceedings{shen2019icra,  
  title={Beyond Photometric Loss for Self-Supervised Ego-Motion Estimation},
  author={Shen, Tianwei and Luo, Zixin and Zhou, Lei and Deng, Hanyu and Zhang, Runze and Fang, Tian and Quan, Long},  
  booktitle={International Conference on Robotics and Automation},  
  year={2019},  
  organization={IEEE}  
}
```

# Environment
This codebase is tested on Ubuntu 16.04 with Tensorflow 1.7 and CUDA 9.0.

# Pre-trained Models
Download the [models](https://drive.google.com/file/d/1xWNm9MclJHD729uS6U6k2Oopn--Vnban/view?usp=sharing) presented in the paper, and then unzip them into the `ckpt` folder under the root.

# Train and Test
See `command.sh` for details, specific instructions will be updated later, together with additional intermediate geometric computations used in this method.

# Contact
Feel free to contact me (Tianwei) if you have any questions, either by email or by issue.

# Acknowledgements
We appreciate the great works/repos along this direction, such as [SfMLearner](https://github.com/tinghuiz/SfMLearner) and [GeoNet](https://github.com/yzcjtr/GeoNet). 