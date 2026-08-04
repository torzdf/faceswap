# Model Porting from Keras -> Torch

## dfaker
- [x]
- [x] 128 learn_mask
- [x] 256
- [x] Legacy handling

## dfl-h128
- [x]
- [x] Standard
- [x] Standard learn_mask
- [x] lowmem
- [x] lowmem learn_mask
- [x] Legacy handling

## dfl-sae
- [x]
- [x] df
- [x] df learn_mask
- [x] df multi_out
- [x] liae learn_mask multi_out
- [x] Legacy handling

## dlight
- [x]
- [x] best good 256
- [x] best good 256 learn_mask  Mask is numerically off, but looks fine. Not worth digging into as niche, and no obvious errors
- [x] fair fast 128 learn_mask
- [x] lowmem fast 384 learn_mask
- [x] Legacy handling

## IAE
- [x]
- [x] standard
- [x] learn_mask
- [x] Legacy handling                   DIFF: [0.0002, 0.0005, 0.0002, 0.0003]

## lightweight
- [x]
- [x] standard
- [x] learn_mask
- [x] Legacy handling

## Original
- [x]
- [x] standard
- [x] low_mem learn_mask
- [x] Legacy handling

## realface
- [x]
- [x] standard
- [x] 64 64 1024 96 544 learn_mask
- [x] Legacy handling

## unbalanced
- [x]
- [x] standard
- [x] low_mem learn_mask
- [x] Legacy handling

## villain
- [x]
- [x] standard
- [x] low_mem learn_mask
- [x] Legacy handling

## phaze-a
### Needs dedicated testing:
    - dec upscales in fc
    - reshaping fc for incorrect output size vs fc dims (make change from legacy and handle full reshape in fc)
- [ ]
- [x] Original style (Lightweight)

### Presets
[x] clip128_lm       WEIGHTS PORT DIFF: [0.0004, 0.0011, 0.0004, 0.0011]
[x] clip256_lm       WEIGHTS PORT DIFF: [0.0001, 0.0036, 0.0002, 0.0016]
[x] clip448_lm       WEIGHTS PORT DIFF: [0.0001, 0.0032, 0.0001, 0.0026]
[x] dfaker_lm        WEIGHTS PORT DIFF: [0.0004, 0.0026, 0.0006, 0.0024]
[x] dny256           WEIGHTS PORT DIFF: [0.0005, 0.0003]
[x] dny512           WEIGHTS PORT DIFF: [0.0004, 0.0006]
[x] dny1024          WEIGHTS PORT DIFF: [0.0004, 0.0002]
[x] h128_lm          WEIGHTS PORT DIFF: [0.0024, 0.0033, 0.0028, 0.0029]
[] iae_lm           WEIGHTS PORT DIFF: [0.0792, 0.1377, 0.0022, 0.003]
[x] lw_lm            WEIGHTS PORT DIFF: [0.0017, 0.002, 0.0017, 0.0017]
[x] orig_lm          WEIGHTS PORT DIFF: [0.0013, 0.0018, 0.0015, 0.0016]
[x] saedf_lm         WEIGHTS PORT DIFF: [0.0001, 0.0031, 0.0001, 0.0016]
[x] saedfhd_lm       WEIGHTS PORT DIFF: [0.0001, 0.0017, 0.0001, 0.002]
[] saeliae_lm       WEIGHTS PORT DIFF: [0.0001, 0.0036, 0.052, 0.0817]
[] saeliaehd_lm     WEIGHTS PORT DIFF: [0.0, 0.0021, 0.0467, 0.0675]
[] stojo_lm         WEIGHTS PORT DIFF: [0.1186, 0.0213, 0.0542, 0.0137]
[] sym384           WEIGHTS PORT DIFF: [0.037, 0.0983]

# Phaze-A Layer testing
## Encoders
[ ]
[x] clipv_farl-b-16-16      WEIGHTS PORT DIFF: [1e-6]  # Weights can't import to pos_embed when res is different between original model + new model # FaRL test only # ZIPPED
[x] clipv_farl-b-16-64      WEIGHTS PORT DIFF: [1e-7]  # Weights can't import to pos_embed when res is different between original model + new model # FaRL test only # ZIPPED
[x] clipv_vit-b-16          WEIGHTS PORT DIFF: [2e-6]  # Weights can't import to pos_embed when res is different between original model + new model # ZIPPED
[x] clipv_vit-b-32          WEIGHTS PORT DIFF: [1e-6]  # Weights can't import to pos_embed when res is different between original model + new model # ZIPPED
[x] clipv_vit-l-14          WEIGHTS PORT DIFF: [2e-6]  # Weights can't import to pos_embed when res is different between original model + new model # ZIPPED
[x] clipv_vit-l-14-336px    WEIGHTS PORT DIFF: [2e-6]  # Weights can't import to pos_embed when res is different between original model + new model # ZIPPED
[x] convnext_base           WEIGHTS PORT DIFF: [1e-4]  # Pre-stem (include_preprocessing) needs to be disabled in Keras for matching => Need to validate this for Torch legacy
[x] convnext_extra_large    WEIGHTS PORT DIFF: [5e-7]  # Pre-stem (include_preprocessing) needs to be disabled in Keras for matching => Need to validate this for Torch legacy
[x] convnext_large          WEIGHTS PORT DIFF: [2e-7]  # Pre-stem (include_preprocessing) needs to be disabled in Keras for matching => Need to validate this for Torch legacy
[x] convnext_small          WEIGHTS PORT DIFF: [5e-7]  # Pre-stem (include_preprocessing) needs to be disabled in Keras for matching => Need to validate this for Torch legacy
[x] convnext_tiny           WEIGHTS PORT DIFF: [1e-5]  # Pre-stem (include_preprocessing) needs to be disabled in Keras for matching => Need to validate this for Torch legacy
[x] densenet121             WEIGHTS PORT DIFF: [3e-5]
[x] densenet169             WEIGHTS PORT DIFF: [3e-5]
[x] densenet201             WEIGHTS PORT DIFF: [4e-5]
[ ] efficientnet_b0         WEIGHTS PORT DIFF: [265.10] - Structures match. This seems to be an accumulation error. Issue starts at very first block1a_project_conv. We need to check this with real output to see if this is an actual issue or not. See test results at bottom of page
[x] efficientnet_b1         WEIGHTS PORT DIFF: [6e-6]
[x] efficientnet_b2         WEIGHTS PORT DIFF: [6e-6]
[x] efficientnet_b3         WEIGHTS PORT DIFF: [7e-6]
[x] efficientnet_b4         WEIGHTS PORT DIFF: [8e-6]  # Have tested that it is correct to remove scaling + norm (it is, stats match at input to first actual EffNet input). However, outputs from full Phaze-A are incorrect. Need to diagnose whether this is EffNet issue or wider PA issue
[x] efficientnet_b5         WEIGHTS PORT DIFF: [1e-5]
[x] efficientnet_b6         WEIGHTS PORT DIFF: [1e-5]
[x] efficientnet_b7         WEIGHTS PORT DIFF: [2e-5]
[ ] efficientnet_v2_b0      WEIGHTS PORT DIFF: [16.80] - Structures match. See v1_b0 issue above
[x] efficientnet_v2_b1      WEIGHTS PORT DIFF: [1e-4]
[x] efficientnet_v2_b2      WEIGHTS PORT DIFF: [1e-4]
[x] efficientnet_v2_b3      WEIGHTS PORT DIFF: [2e-4]
[ ] efficientnet_v2_l       WEIGHTS PORT DIFF: [10840668.0] - This blows out + grows around 4/5 boundary. No idea why. Weight mapping is definitely correct. Model args are definitely correct. See stats below
[x] efficientnet_v2_m       WEIGHTS PORT DIFF: [3e-4]
[x] efficientnet_v2_s       WEIGHTS PORT DIFF: [4e-4]
[x] fs_original
[x] fs_original_alt
[x] inception_resnet_v2     WEIGHTS PORT DIFF: [3e-5]   DIFF: [0.0002, 0.0001]
[ ] inception_v3            WEIGHTS PORT DIFF: [1e-5]   DIFF: [0.0001, 0.0001]  !! BUG IN TORCH WHEN LOADING IMAGENET WEIGHTS aux_logits MUST BE ``True`` BUT WE DON'T WANT IT. LOOK AT INIT_WEIGHTS WARNING.
[x] mobilenet               WEIGHTS PORT DIFF: [7e-5]
[x] mobilenet_v2            WEIGHTS PORT DIFF: [2e-6]
[x] mobilenet_v3_large      WEIGHTS PORT DIFF: [5e-7]
[x] mobilenet_v3_large(min) WEIGHTS PORT DIFF: [4e-7]  # needs weights
[x] mobilenet_v3_small      WEIGHTS PORT DIFF: [9e-7]
[x] mobilenet_v3_small(min) WEIGHTS PORT DIFF: [8e-7]  # needs weights
[ ] nasnet_large            WEIGHTS PORT DIFF: [23.87]  DIFF: [0.5763, 0.5737]
[ ] nasnet_mobile           WEIGHTS PORT DIFF: [18.77]  DIFF: [0.5763, 0.5737]  # NEEDS FSK2 ENV TO RUN KERAS VERS
[x] resnet50                WEIGHTS PORT DIFF: [8e-5]   DIFF: [0.0002, 0.0002]
[x] resnet50_v2             WEIGHTS PORT DIFF: [1e-4]   DIFF: [0.0002, 0.0002]
[x] resnet101               WEIGHTS PORT DIFF: [2e-4]   DIFF: [0.0004, 0.0004]
[x] resnet101_v2            WEIGHTS PORT DIFF: [2e-4]   DIFF: [0.0004, 0.0003]
[x] resnet152               WEIGHTS PORT DIFF: [4e-4]   DIFF: [0.0005, 0.0005]
[x] resnet152_v2            WEIGHTS PORT DIFF: [6e-4]   DIFF: [0.0004, 0.0004]
[x] vgg16                   WEIGHTS PORT DIFF: [0.0]    DIFF: [0.0009, 0.0008]
[x] vgg19                   WEIGHTS PORT DIFF: [0.0]    DIFF: [0.0006, 0.0006]
[x] xception                WEIGHTS PORT DIFF: [1e-5]   DIFF: [0.0, 0.0]
[x] bottleneck in encoder
## Bottleneck
[x]
[x] dense
[x] flatten
[x] max pool
[x] avg pool
[x] none norm
[x] instance norm
[x] layer norm
[x] rms norm
## FC
[ ]
[x] depth 1-4
[x] min_filters 64-256
[x] max_filters 512-1536
[x] dimensions 2-7
[x] filter_slope (-0.75, -.66, -0.5, -0.33, -.25, 0, .25, .33, .5, 0.66, .75)
[x] dropout
[x] No upscale
[x] upsample2D (1-3)
[x] subpixel (1-3)
[x] hybrid (1-3)
[x] fast (1-3)
[x] resize (1-3)
[x] upsample_filts 64-512
[x] bottleneck in fc
[ ] upscales in fc 1-3
## Inter-GBlock
[x]
[x] depth 1-4
[x] min_filters 64-512
[x] max_filters 128-768
[x] filter_slope (-0.75, -.66, -0.5, -0.33, -.25, 0, .25, .33, .5, 0.66, .75)
[x] dropout
[x] bottleneck in fc
## Gblock
[x] Structure and default output
## Decoders
[x]
[x] learn_mask
[x] Subpixel
[x] DNY
[x] hybrid
[x] fast
[x] resize
[x] none norm
[x] instance norm
[x] layer norm
[x] group norm
[x] rms norm
[x] batch norm
[x] Force dimension reshape
[x] min_filts 32-384
[x] max_filts 256-2048
[x] out_size 64-2048
[x] slope mode full
[x] slope mode cap_min
[x] slope mode cap_max
[x] filter_slope (-0.75, -.66, -0.5, -0.33, -.25, 0, .25, .33, .5, 0.66, .75)
[x] res_blocks (0, 1, 2, 3, 4)
[x] slip_last_res
[x] gaussian
[x] out_kernel 1-9



EFFNET_B0 Hook Tests:
b0 hook test:
KerasLayer@TorchPath@KShape@TShape@MaxDiff@KMin@KMax@TMin@TMax
block1a_dwconv@efficientnetB0.1.0.block.0.0@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000003@-8.1221@8.6411@-8.1221@8.6411
block1a_bn@efficientnetB0.1.0.block.0.1@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000010@-24.3650@30.0130@-24.3650@30.0130
block1a_activation@efficientnetB0.1.0.block.0.2@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000010@-0.2785@30.0130@-0.2785@30.0130
block1a_se_reshape@efficientnetB0.1.0.block.1.avgpool@(1, 1, 1, 32)@(1, 1, 1, 32)@0.000000@-0.2136@2.2980@-0.2136@2.2980
block1a_se_reduce@efficientnetB0.1.0.block.1.activation@(1, 1, 1, 8)@(1, 1, 1, 8)@0.000000@-0.2767@2.1608@-0.2767@2.1608
block1a_se_expand@efficientnetB0.1.0.block.1.scale_activation@(1, 1, 1, 32)@(1, 1, 1, 32)@0.000000@0.2152@0.7504@0.2152@0.7504
block1a_project_conv@efficientnetB0.1.0.block.2.0@(1, 112, 112, 16)@(1, 112, 112, 16)@0.003036@-18.0783@11.9626@-18.0783@11.9626
block1a_project_bn@efficientnetB0.1.0.block.2.1@(1, 112, 112, 16)@(1, 112, 112, 16)@0.007133@-28.1634@25.9071@-28.1634@25.9071
block2b_project_bn@efficientnetB0.2.1.block.3.1@(1, 56, 56, 24)@(1, 56, 56, 24)@0.014949@-38.5090@37.2441@-38.5090@37.2441
block3b_project_bn@efficientnetB0.3.1.block.3.1@(1, 28, 28, 40)@(1, 28, 28, 40)@0.047044@-66.5446@68.2859@-66.5434@68.2955
block4c_project_bn@efficientnetB0.4.2.block.3.1@(1, 14, 14, 80)@(1, 14, 14, 80)@0.248325@-159.4065@161.2227@-159.4295@161.1709
block5c_project_bn@efficientnetB0.5.2.block.3.1@(1, 14, 14, 112)@(1, 14, 14, 112)@1.170216@-297.7014@323.8462@-298.2259@323.6599
block6d_project_bn@efficientnetB0.6.3.block.3.1@(1, 7, 7, 192)@(1, 7, 7, 192)@69.710693@-914.8705@915.2687@-906.4109@907.5565
block7a_project_bn@efficientnetB0.7.0.block.3.1@(1, 7, 7, 320)@(1, 7, 7, 320)@177.448792@-1140.0803@1143.9122@-1149.2787@1150.7732
top_activation@efficientnetB0.8.2@(1, 7, 7, 1280)@(1, 7, 7, 1280)@265.098175@-0.2785@1217.2451@-0.2784@1224.7277

b1 hook test:
KerasLayer@TorchPath@KShape@TShape@MaxDiff@KMin@KMax@TMin@TMax
block1a_dwconv@efficientnetB1.1.0.block.0.0@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000000@-0.6843@0.5510@-0.6843@0.5510
block1a_bn@efficientnetB1.1.0.block.0.1@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000000@-0.6830@0.5538@-0.6830@0.5538
block1a_activation@efficientnetB1.1.0.block.0.2@(1, 112, 112, 32)@(1, 112, 112, 32)@0.000000@-0.2292@0.3517@-0.2292@0.3517
block1a_se_reshape@efficientnetB1.1.0.block.1.avgpool@(1, 1, 1, 32)@(1, 1, 1, 32)@0.000000@-0.1001@0.1658@-0.1001@0.1658
block1a_se_reduce@efficientnetB1.1.0.block.1.activation@(1, 1, 1, 8)@(1, 1, 1, 8)@0.000000@-0.1075@0.0957@-0.1075@0.0957
block1a_se_expand@efficientnetB1.1.0.block.1.scale_activation@(1, 1, 1, 32)@(1, 1, 1, 32)@0.000000@0.4784@0.5232@0.4784@0.5232
block1a_project_conv@efficientnetB1.1.0.block.2.0@(1, 112, 112, 16)@(1, 112, 112, 16)@0.000034@-0.3189@0.2983@-0.3189@0.2983
block1a_project_bn@efficientnetB1.1.0.block.2.1@(1, 112, 112, 16)@(1, 112, 112, 16)@0.000035@-0.3184@0.2968@-0.3184@0.2968
block1b_project_bn@efficientnetB1.1.1.block.2.1@(1, 112, 112, 16)@(1, 112, 112, 16)@0.000008@-0.1178@0.1851@-0.1178@0.1851
block2c_project_bn@efficientnetB1.2.2.block.3.1@(1, 56, 56, 24)@(1, 56, 56, 24)@0.000015@-0.0339@0.0419@-0.0339@0.0419
block3c_project_bn@efficientnetB1.3.2.block.3.1@(1, 28, 28, 40)@(1, 28, 28, 40)@0.000010@-0.0257@0.0242@-0.0257@0.0242
block4d_project_bn@efficientnetB1.4.3.block.3.1@(1, 14, 14, 80)@(1, 14, 14, 80)@0.000009@-0.0262@0.0312@-0.0262@0.0312
block5d_project_bn@efficientnetB1.5.3.block.3.1@(1, 14, 14, 112)@(1, 14, 14, 112)@0.000008@-0.0240@0.0211@-0.0240@0.0211
block6e_project_bn@efficientnetB1.6.4.block.3.1@(1, 7, 7, 192)@(1, 7, 7, 192)@0.000008@-0.0312@0.0302@-0.0312@0.0302
block7b_project_bn@efficientnetB1.7.1.block.3.1@(1, 7, 7, 320)@(1, 7, 7, 320)@0.000006@-0.0228@0.0246@-0.0228@0.0246
top_activation@efficientnetB1.8.2@(1, 7, 7, 1280)@(1, 7, 7, 1280)@0.000006@-0.0149@0.0153@-0.0148@0.0153



v2_l hook_test:
KerasLayer	TorchPath	KShape	TShape	MaxDiff	KMin	KMax	TMin	TMax
stem_conv	efficientnetV2L.0.0	(1, 112, 112, 32)	(1, 112, 112, 32)	0	-1.1432	1.1951	-1.1432	1.1951
stem_bn	efficientnetV2L.0.1	(1, 112, 112, 32)	(1, 112, 112, 32)	0	-1.2525	1.3714	-1.2525	1.3714
stem_activation	efficientnetV2L.0.2	(1, 112, 112, 32)	(1, 112, 112, 32)	0	-0.2784	1.0939	-0.2784	1.0939
block1a_project_conv	efficientnetV2L.1.0.block.0.0	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000045	-1.1585	1.007	-1.1585	1.007
block1a_project_bn	efficientnetV2L.1.0.block.0.1	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000047	-1.1463	0.9928	-1.1463	0.9928
block1a_project_activation	efficientnetV2L.1.0.block.0.2	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000029	-0.2764	0.7244	-0.2764	0.7244
block1b_project_activation	efficientnetV2L.1.1.block.0.2	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000056	-0.2783	0.9958	-0.2783	0.9958
block1c_project_activation	efficientnetV2L.1.2.block.0.2	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000138	-0.2785	1.3409	-0.2785	1.3409
block1d_project_activation	efficientnetV2L.1.3.block.0.2	(1, 112, 112, 32)	(1, 112, 112, 32)	0.000169	-0.2785	1.3784	-0.2785	1.3784
block2g_project_bn	efficientnetV2L.2.6.block.1.1	(1, 56, 56, 64)	(1, 56, 56, 64)	0.002368	-2.9764	2.8707	-2.9759	2.8711
block3g_project_bn	efficientnetV2L.3.6.block.1.1	(1, 28, 28, 96)	(1, 28, 28, 96)	0.022051	-20.9403	18.8371	-20.9379	18.8368
block4a_project_bn	efficientnetV2L.4.0.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.020507	-14.6453	15.3239	-14.6401	15.3241
block4b_project_bn	efficientnetV2L.4.1.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.019061	-9.3673	12.8398	-9.3674	12.8381
block4c_project_bn	efficientnetV2L.4.2.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.027409	-14.9666	13.8847	-14.9664	13.8933
block4d_project_bn	efficientnetV2L.4.3.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.03011	-19.7585	17.6849	-19.7506	17.6765
block4e_project_bn	efficientnetV2L.4.4.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.041873	-22.9466	26.3744	-22.9582	26.3725
block4f_project_bn	efficientnetV2L.4.5.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.049974	-30.2848	30.4924	-30.2866	30.4905
block4g_project_bn	efficientnetV2L.4.6.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.074184	-45.6567	39.0201	-45.6663	38.9986
block4h_project_bn	efficientnetV2L.4.7.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.087964	-64.9929	53.3172	-64.969	53.3221
block4i_project_bn	efficientnetV2L.4.8.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.132837	-76.615	85.9638	-76.5737	85.9304
block4j_project_bn	efficientnetV2L.4.9.block.3.1	(1, 14, 14, 192)	(1, 14, 14, 192)	0.161461	-103.9344	91.6483	-103.9242	91.6771
block5a_project_bn	efficientnetV2L.5.0.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	0.21993	-112.7477	116.5043	-112.7832	116.5586
block5b_project_bn	efficientnetV2L.5.1.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	0.254349	-133.3698	118.9295	-133.3717	118.9164
block5c_project_bn	efficientnetV2L.5.2.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	0.545982	-177.1564	196.4384	-177.1478	196.6042
block5d_project_bn	efficientnetV2L.5.3.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	1.066055	-264.1185	264.6018	-263.7057	264.4379
block5e_project_bn	efficientnetV2L.5.4.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	2.656448	-421.9317	415.9886	-422.6145	415.2366
block5f_project_bn	efficientnetV2L.5.5.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	4.869568	-449.5712	518.4721	-448.3813	514.3027
block5g_project_bn	efficientnetV2L.5.6.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	8.242828	-624.4092	825.7848	-626.4436	825.0746
block5h_project_bn	efficientnetV2L.5.7.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	14.324532	-1001.6989	748.2686	-995.1965	744.1859
block5i_project_bn	efficientnetV2L.5.8.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	31.544189	-906.0367	793.3534	-881.7699	794.0597
block5j_project_bn	efficientnetV2L.5.9.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	78.78595	-1223.7461	1550.5543	-1221.3074	1543.9799
block5s_project_bn	efficientnetV2L.5.18.block.3.1	(1, 14, 14, 224)	(1, 14, 14, 224)	4648.561523	-7983.9976	7372.2563	-8013.6851	7574.5996
block6y_project_bn	efficientnetV2L.6.24.block.3.1	(1, 7, 7, 384)	(1, 7, 7, 384)	3591139	-2635533.5	2987031.75	-2262132	2145634.75
block7g_project_bn	efficientnetV2L.7.6.block.3.1	(1, 7, 7, 640)	(1, 7, 7, 640)	13060014	-13209879	14252231	-12424061	12732645
top_activation	efficientnetV2L.8.2	(1, 7, 7, 1280)	(1, 7, 7, 1280)	10840668	0	13460824	0	11018882
