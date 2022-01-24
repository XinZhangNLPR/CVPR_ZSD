rlaunch --gpu=8 --cpu=24 --memory=150000 -- ./tools/dist_test.sh configs/listmle_rank_zsi/48_17/test/gzsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_decoder_gzsi.py work_dirs/listmle_rank_zsi/48_17/epoch_12.pth 8 --json_out results/listmle_gzsi__48_17.json



rlaunch --gpu=8 --cpu=24 --memory=150000 -- ./tools/dist_test.sh configs/listmle_rank_zsi/48_17/test/zsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_decoder.py work_dirs/listmle_rank_zsi/48_17/epoch_12.pth 8 --json_out results/listmlezsi_48_17.json
