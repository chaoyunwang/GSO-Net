# GSO-Net: Grid Surface Optimization via Learning Geometric Constraints

This is the implementation code of AAAI2024 paper GSO-Net: Grid Surface Optimization via Learning Geometric Constraints. The project page can be seen here: [Project](https://chaoyunwang.github.io/GSO-Net/).

## Requirement

The experiment uses NvidIa 3080 graphics card and installs the following environment:

- python
- torch
- numpy
- pywavefront
- scipy
- tensorboard

## Download

The structure of the material folder related to the reproduction Code is as follows, where the Code is this repository.

```plain
├─GSO-Net
  ├─Dataset
  ├─Task-pretrained_model
  ├─Task-test_result
  ├─Figure-file
  └─Code
```

### Dataset

The grid surface dataset used in this paper is divided into trainset and testset, and saved in npy format, which can be obtained here: [Dataset](https://pan.baidu.com/s/1OOdm67qXSby_satm2XFA4w?pwd=c94n)(1GB).

```plain
├─Dataset
  ├─train
  └─test
```

### Task-pretrained_model

The pretrained model corresponding to each experiment in the paper are provided here: [Task-pretrained_model](http)

```plain
├─Task-pretrained_model
  ├─Developable
  │  ├─Developable_Net-S.pth
  │  ├─Developable_Net-C.pth
  │  └─Developable_Net-F.pth
  ├─Denoise
  │  ├─Denoise_0.001.pth
  │  ├─Denoise_0.005.pth
  │  ├─Denoise_0.010.pth
  │  └─Denoise_0.015.pth
  └─Flatten
     ├─Flatten_Net.pth
     └─Flatten_Net.pth
```

### Task-test_result

The test result  corresponding to each experiment in the paper are provided here: [Task-test_result](http)(2GB).

```plain
├─Task-test_result
  ├─Developable
  │  ├─Developable_Net-C
  │  ├─Developable_Net-CF
  │  ├─Developable_Net-S 
  │  └─Developable_TNO
  ├─Denoise
  │  ├─evaluate_testset
  │  │  ├─noise-0.001
  │  │  ├─noise-0.005
  │  │  ├─noise-0.010
  │  │  └─noise-0.015
  │  └─test_result
  │      ├─noise-0.001
  │      ├─noise-0.005
  │      ├─noise-0.010
  │      └─noise-0.015
  └─Flatten
     ├─Flatten_Init
     ├─Flatten_Net
     ├─Flatten_Net-W
     └─Flatten_TNO
```

### Figure-file

The source file corresponding to the surface optimization image involved in the paper and supplementary materials, can be available here: [Figure-file ](http)

```plain
├─Figure-file
  ├─paper
  │  ├─Figure7
  │  ├─Figure8
  │  └─Figure9
  └─supplementary
     ├─Figure4
     ├─Figure5
     ├─Figure6
     └─Figure7
```

## Bezier surface

You can adjust the Parameters in program "`bezier_dataset.py`"  to generate Bezier surfaces with diverse features, including the calculation of surface features. The file name is named with the surface features, and the data is saved in npy format. The dataset used in this article is generated in this way.

```plain
python bezier_dataset.py
```

## Convert Example

The conversion between the grid surface obj file and the npy file. The corresponding npy file and obj file are stored in the "./opt_example" folder.  obj file can be used the [meshlab ](https://www.meshlab.net/)software to open. Conversion cmmand is：

```python
python obj2npy.py
```

## Inference Example

 The "opt_example" folder contains files for Inference. Files about the pretrained models can be found in the "Task-pretrained_model" directory.  

### Developable

```plain
python inference.py --task Developable --model_path ../Task-pretrained_model/Developable/Developable_Net-C.pth --input_obj ./opt_example/Developable.obj --output_obj ./opt_example/opt-Developable.obj
```

### Flatten

```plain
python inference.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net-W.pth --input_obj ./opt_example/Flatten.obj --output_obj ./opt_example/opt-Flatten.obj
```

### Denoise

```plain
python inference.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.010.pth --input_obj ./opt_example/Denoise.obj --output_obj ./opt_example/opt-Denoise.obj
```

## Train

 For the tasks of optimizing developable surfaces and surface denoising , as described in the supplementary material, we conducted a two-stage training.

### Developable

\1. Pre-training phase 

A self-encoder network weight "pretrained_auto_en-de.pth" is obtained. Set the weight coefficient in "nntools.py", `weight_2 = 1.0,weight_4 = 0.0 `, run the following command:

```plain
python main.py --task Developable --root_dir ../Dataset  --output_dir ./pretrained/ --num_epochs 1000 --lr 1e-3 --batch_size 256
```

\2. Optimize Gaussian curvature based on pre-training model 

Set the loss weight in "nntools.py", `weight_2 = 1.0 * np. exp (-a * (epoch)/4000.0), weight_4 = 1.0 `, where a is the weight decay coefficient, Net-C and Net-F are set to 3 in the article, and 4 in the Net-S, run the command:

```plain
python main.py --task Developable --root_dir ../Dataset --pretrain_model ./pretrained/pretrained_auto_en-de.pth --output_dir ../Developable/ --num_epochs 4000 --lr 1e-4 --batch_size 256
```

\3. The Net-F model in the paper is a model that is retrained by using Dataset_opt2 on the dataset optimized by the Net-C model. The weight setting is consistent with the Net-C. The Dataset_opt2 dataset is obtained as follows:

```plain
python construct_optdataset.py
```

The training command is as follows:

```plain
python main.py --task Developable --root_dir ../Dataset_opt2 --pretrain_model ./pretrained/pretrained_auto_en-de.pth --output_dir ../Developable/ --num_epochs 4000 --lr 1e-4 --batch_size 256
```

### Flatten

In the surface flattening task, the network output is 2d. You can change the loss2  function in nntools.py, corresponding to the Net and Net-W models, respectively. `loss2 = self. loss_criterion.criterion_2d_3d_pow2_dim2_gaussweight (x, y)#criterion_2d_3d_pow2_dim2, criterion_2d_3d_pow2_dim2_gaussweight `,

run the following command to train:

```plain
python main.py --task Flatten --root_dir ../Dataset --output_dir ../Flatten/ --num_epochs 1000 --lr 1e-4 --batch_size 256
```

### Denoise

In the surface noise removal task, you can adjust the noise intensity during training by changing the "scale" parameter in "data.py": `noise = np. random. normal (loc = 0, scale = 0.015, size = noisy. shape)#add noise `, then you can use the pre-training model for secondary optimization fairness loss:

```plain
python main.py --task Denoise --root_dir ../Dataset --pretrain_model ./pretrained/pretrained_auto_en-de.pth --output_dir ../Denoise/ --num_epochs 1000 --lr 1e-4 --batch_size 256
```

## Test

Use the trained model to optimize the data in the testset.

### Developable

For the developable surface optimization task, run the following command:

Test for the Net-S model:

```plain
python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-S.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Developable/Developable_Net-S
```

Test for the Net-C model:

```plain
python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-C.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Developable/Developable_Net-C
```

Net-CF is the results by optimized using Net-C model and then Net-F model:

```plain
python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-F.pth --input_dir ../Task-test_result/Developable/Developable_Net-C --output_dir ../Task-test_result/Developable/Developable_Net-CF
```

### Flatten

Run the following command on the surface flattening task:

Task for the Flatten_Net model:

```plain
python test.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Flatten/Flatten_Net
```

Task for the Flatten_Net-W model:

```plain
python test.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net-W.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Flatten/Flatten_Net-W
```

### Denoise

For the surface denoise task, we generate a fixed noise surface testset,  the "evaluate_testset" in "Task-test_result" is generated by adding noise to the testset. Set parameters `noise = np. random. normal (loc = 0, scale = level, size = noisy. shape) `, level is the corresponding noise intensity parameter:

```plain
python construct_noise_dataset.py
```

Run the following command to obtain the corresponding test results.

Task for the Denoise_0.001 model:

```plain
python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.001.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.001 --output_dir ../Task-test_result/Denoise/test_result/noise-0.001
```

Task for the Denoise_0.005 model:

```plain
python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.005.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.005 --output_dir ../Task-test_result/Denoise/test_result/noise-0.005
```

Task for the Denoise_0.010 model:

```plain
python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.010.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.010 --output_dir ../Task-test_result/Denoise/test_result/noise-0.010
```

Task for the Denoise_0.015 model:

```plain
python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.015.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.015 --output_dir ../Task-test_result/Denoise/test_result/noise-0.015
```

## Evaluate Metrics

Run the evaluate_metric.py program to evaluate the optimization results of the test set. You can select different task codes and change the path to calculate the optimization result indicators:

```plain
python evaluate_metric.py
```

## Ciation

If the code is useful for your research, please consider citing:

```plain
@inproceedings{wang2024GSO-Net,
  title={GSO-Net: Grid Surface Optimization via Learning Geometric Constraints},
  author={Chaoyun Wang, Jingmin Xin, Nanning Zheng, Caigui Jiang},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  year={2024}
}
```
