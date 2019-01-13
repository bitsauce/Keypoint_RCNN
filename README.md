# Keypoint R-CNN

This project extends the instance-level mask preditions of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) to enable keypoint estimation.

Here is a figure of the network architecture:

![Model Overview](doc/model_overview.png)

The architecure of the heads are as follows:

![Heads](doc/model_heads.png)

This network is trained on MS COCO with its keypoint annotations. Below is an example of such an annotation:

![MS COCO Example](doc/gt_example.png)

After training for 40k iterations, the model predicts masks such these:

![Prediction Examples](doc/predicted_kps_examples.png)

Here I have visualized the heatmaps of inset (e):

![Heatmaps](doc/prediction_kp_heatmap_2063.png)

I ran the evaluation from https://github.com/matteorr/coco-analyze, and it gave me the following:

Precision/Recall Curve:

![Precision/Recall Curve](doc/analyze_prc_[pose_rcnn][all][20]-1.png)

Overall Keypoint Errors:

![Overall Keypoint Errors](doc/analyze_overall_keypoint_errors-1.png)

AP Improvement Areas:

![AP Improvement Areas](doc/analyze_ap_improv_areas_all-1.png)

Per Keypoint Error Breakdown:

![Per Keypoint Error Breakdown](doc/analyze_keypoint_breakdown-1.png)
