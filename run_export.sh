## quick script to run export and evaluation

export_folder='superpoint_coco_heat_fl_nms4_det0.015'
# export_folder='superpoint_kitti_heat2_0'
echo $export_folder
# python3 export.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
python3 evaluation.py /mnt/disks/user/project/logs/$export_folder/predictions --repeatibility --homography --outputImg --plotMatching

conda create --name py36-sp python=3.6
