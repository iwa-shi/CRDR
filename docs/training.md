# Training

### 1. Dataset Preparation

#### Training dataset

For training, we used [OpenImage](https://storage.googleapis.com/openimages/web/index.html) dataset.
Specifically, we used subset `train_0` ~ `train_9` (~1.1M images), which can be downloaded from [here](https://github.com/cvdfoundation/open-images-dataset?tab=readme-ov-file#download-images-with-bounding-boxes-annotations).
Place each sub-directory (`train_X`) in a main directory:
```
openimage/
├── train_0/
│   ├── XXXXXX.jpg
│   ├── ...
│   └── YYYYYY.jpg
├── train_1/
├── ...
└── train_9/

```

> [!NOTE]
> If you can't download 10 sub-directories of OpenImage, you can download some of them and modify the `subset_list` key in the config accordingly.
> For example, change `subset_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` to `subset_list: [0, 1]` if you have sub-directories `train_0` and `train_1`.


#### Validation dataset

For validation, we used [Kodak](https://r0k.us/graphics/kodak/) dataset. The dataset directory should be structured as follows:
```
kodak/
├── kodim01.png
├── kodim02.png
├── ...
└── kodim24.png
```

After that, specify the path of the datasets in the root_dir key in [config/_base_/dataset/openimage_kodak.yaml](../config/_base_/dataset/openimage_kodak.yaml).


### 2. Run Training

Training has 3 stages and a total of 5 million iterations, as shown below:

|  | Training | Loss | Iterations | Config file | Pre-trained model |
|-|-|-|-|-|-|
| Stage 1 | Single-rate High-bpp Training (Warmup) | Rate, MSE, LPIPS | 1M | [crdr_stage_1.yaml](../config/crdr_stage_1.yaml) | -
| Stage 2 | Multi-rate w/o GAN Training | Rate, MSE, LPIPS | 1M | [crdr_stage_2.yaml](../config/crdr_stage_2.yaml) | [GDrive](https://drive.google.com/file/d/1UY-swZ-ZjUulVGI8TCkkmqHpnWNFLi37/view?usp=sharing)
| Stage 3 | Multi-rate-realism Training | Rate, MSE, LPIPS, Adversarial | 3M | [crdr_stage_3.yaml](../config/crdr_stage_3.yaml) | [GDrive](https://drive.google.com/file/d/1H6T9-k0RX5SXk0VljHNiXUGXZrZl2seb/view?usp=drive_link)

For each stage, you can run the training with the following command:
```
poetry run python ./scripts/train.py {CONFIG_PATH} -d {DEVICE}
```
#### Other options:
- `-wb, --use_wandb`: Enable Weights & Biases logging. See `init_wandb()` in `src/trainer/base_trainer.py`
- `--debug`: Run training in debug mode.
- `-si, --start_iter`: Resume training from a specific iteration.
- `-dr, --dry_run`: Print the model and exit.
See `src/utils/options.py` for more options.

You can use our pre-trained stage-2 model to train your own stage-3 model. To do so, you may need to modify `pretrained_weight_path` in the config file.

#### Example:
```
poetry run python ./scripts/train.py ./config/crdr_stage_2.yaml -d cuda:1 -wb
```

The filename of the config file without the file extension (e.g., `crdr_stage_2`) will be the name of the experiment. This name will be the directory name of the checkpoint. In the above example, a directory `./checkpoint/crdr_stage_2` will be created, and files such as logs, model weights, and validation results will be stored there as follows:

```
checkpoint/
└── crdr_stage_2/
    ├── model/
    │   ├── comp_model_iter{ITER}.pth.tar
    │   └── training_state_iter{ITER}.pth.tar
    ├── config.yaml
    ├── eval_result.csv
    ├── log_loss.csv
    └── train_{TIMESTAMP}.log
```

#### Reproduced Training Log
We have conducted reproduced training for stage-3 by fine-tuning our [stage-2 model](https://drive.google.com/file/d/1UY-swZ-ZjUulVGI8TCkkmqHpnWNFLi37/view?usp=sharing).
You can find the log at [CRDR stage-3 reproduced training log](https://api.wandb.ai/links/shoiwai/l9rfncb1).
The final results of the reproduced model on CLIC can be found [here](../rd_results/CLIC_reproduce.csv).
We obtained similar (or even slightly better at some rate points) results compared to those reported in the paper.

### 3. Test Trained Model

#### Make Reconstructions

You can create reconstructions using the trained model with the following command:
```
poetry run python scripts/compress.py --config_path {CONFIG_PATH} --model_path {MODEL_PATH} --img_dir {PATH/TO/DATASET} --save_dir {SAVE_DIR} -q QUALITY -b BETA --decompress -d cuda
```

For the single-rate model, `--quality (-q)` should be `< 0` because single-rate models don't take the quality (`rate_ind`) parameter (i.e., stage 1 model).
Similarly, `--beta (-b)` should be `< 0` for the single-realism model (i.e., stage 1 and stage 2 models).

##### _Example 1_

Using `-q -1` and `-b -1` for the single-rate model.
```
poetry run python scripts/compress.py --config_path ./config/crdr_stage_1.yaml --model_path ./checkpoint/crdr_stage_1/model/comp_model_iter1000K.pth.tar --img_dir ./dataset/kodak --save_dir ./crdr_results/crdr_stage_1/kodak -q -1 -b -1 --decompress -d cuda
```

##### _Example 2_
Using `-q 1.5` and `-b -1` for the Multi-rate & Single-realism model.
```
poetry run python scripts/compress.py --config_path ./config/crdr_stage_2.yaml --model_path ./checkpoint/crdr_stage_2/model/comp_model_iter1000K.pth.tar --img_dir ./dataset/kodak --save_dir ./crdr_results/crdr_stage_2/kodak -q 1.5 -b -1 --decompress -d cuda
```

#### Calculate Metrics

You can calculate metrics (PSNR, FID, LPIPS, DISTS) of the reconstructions by running the following script:
```
poetry run python scripts/calc_metrics.py --real_dir {PATH/TO/DATASET} --fake_dir {PATH/TO/RECONSTRUCTIONS} -d cuda
```
Results will be stored in `{fake_dir}/_metrics.json`.

___

### Customize Config

Our codebase borrows _Config_ and _Registry_ from [MMCV 1.X](https://github.com/open-mmlab/mmcv/tree/1.x) (the latest MMCV library is 2.X).
Please refer to [Config Tutorial](https://github.com/open-mmlab/mmcv/blob/ff39b4fca8aa18a0fc807d5c5eb1cc3a5c087451/docs/en/understand_mmcv/config.md) and [Registry Tutorial](https://github.com/open-mmlab/mmcv/blob/1.x/docs/en/understand_mmcv/registry.md) to learn the basic usage.

You can customize config files to train your own model. See `config/examples` for some examples, such as changing quality levels, trainer, and discriminator architecture.

Additionally, we provide config files for the version without Charm (Channel-Autoregressive Model), such as `config/_base_/model/elic_hyperprior.yaml`, for faster training and inference. These w/o Charm models are used in ablation studies in our paper.


If you have any questions or encounterd any issues, please feel free to open issue!