# DAE_Keras

## Environment
* tensorflow 1.5
* Keras >= 2.0.8

## Usage
```
python train.py x_dir y_dir 
                [--width WIDTH] [--height HEIGHT] [--channel CHANNEL]
                [--batch_size BATCH_SIZE] [--nb_epoch NB_EPOCH]
                [--nb_sample NB_SAMPLE] [--param_dir PARAM_DIR]
                [--color COLOR]
```
* x_dir : Noisy Image Dir
* y_dir : Original Image Dir

対応するデータをx_dirとy_dir内にそれぞれ同名で配置すること

```
python predict.py model_path param_path image_dir 
                  [--result_dir RESULT_DIR] [--width WIDTH]
                  [--height HEIGHT] [--channel CHANNEL]
                  [--batch_size BATCH_SIZE] [--color COLOR]

```
