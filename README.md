### diffusion training 

```python
python main.py --base configs/latent-diffusion/histo-ldm-kl-8-512-skin.yaml -t --gpus 0,1,2,3
```


### classifier training

```python
CUDA_VISIBLE_DEVICES=7 python classifier_train.py --data_dir /gpuhome/jxy225/PathologyDatasets/stain_transfer/HE_Staining_Variation/train_subsets \
                                    --val_data_dir /gpuhome/jxy225/PathologyDatasets/stain_transfer/HE_Staining_Variation/val_subsets \
                                    --iterations 300000 \
                                    --anneal_lr True \
                                    --batch_size 32 \
                                    --lr 5e-5 \
                                    --save_interval 100 \
                                    --weight_decay 0.05 \
                                    --image_size 512 \
                                    --classifier_attention_resolutions 32,16 \
                                    --classifier_depth 2 \
                                    --classifier_width 128 \
                                     --classifier_pool attention \
                                    --classifier_resblock_updown False \
                                    --classifier_use_scale_shift_norm False
```


### DDIM Example sampling

```python
python example_sampling/skin.py
```
