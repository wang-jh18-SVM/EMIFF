# 文件说明

1. 复现：
   1. [Installation](docs/en/install.md)
      1. 部分依赖项可能存在冲突，可以参考我的[环境配置](wjh.yaml)
   2. [Prepare Dataset](docs/en/prepare_dataset.md)
      1. 已将处理后的kitti格式协同SPD数据集压缩包上传至百度云
      2. 若希望从头处理，注意EMIFF中的dair_vic2kitti_2.py文件是针对DAIR-V2X-C数据集的，与SPD有区别。需要使用DAIR-V2X[github仓库](https://github.com/AIR-THU/DAIR-V2X/blob/main/docs/get_started_spd.md)中代码转换
   3. [Train and Test](docs/en/train_test.md)
      1. 配置文件在cfgs/vic文件夹，主要使用cfgs/vic/vimi_960x540_12e_bs2.py
2. Debug经验
   1. 主要难点在于确认外参读取&转换，可参考：
      1. tools/data_converter/dair_kitti_data_utils.py中get_kitti_image_info函数：从数据集原文件中读取内参，保存到info.pkl
      2. mmdet3d/datasets/dair_vic_dataset.py中get_data_info函数：从info.pkl中读取内参，加载入dataloader
   2. 可使用tools/misc/browse_dataset.py可视化数据集，确认外参是否正确
      1. 可视化过程会调用config文件中的evaluate pipeline，所以删除了其中的normalize环节
   3. 训练得到pth权重后，可使用tools/dist_test.sh文件进行评测，注意修改其中的--out参数路径，保存评测结果pkl文件
      1. 可进一步使用tools/misc/visualize_results_dair.py文件可视化评测结果
   4. 可利用VS CODE自带的调试模式，相关配置在.vscode/launch.json
3. 当前问题
   1. 同样的模型配置，在DAIR-V2X-C数据集上训练精度是Seq-SPD的3倍
      1. 已进行的无效尝试
         1. 可视化数据集、可视化推理结果等，验证外参转换应该是正确的
         2. 调整模型超参数（学习率等），精度都远低于V2X-C结果
         3. 调整输入图像分辨率，不再对输入图像做resize
            1. 原本配置的pipeline中将输入图像resize为(960, 540)，虽然可以训练，但是会导致可视化数据集和推理结果时检测框和图像内容不对应，因为相机的内参是按照原始图像分辨率(1920, 1080)标定的。所以尝试保留原始输入图像分辨率做训练
            2. 但是，在DAIR-V2X-C数据集上训练时，也是resize为(960, 540)进行的训练，并没有影响精度
      2. 推荐可以继续尝试的方向
         1. 检查在DAIR-V2X-C数据集上训练时，代码中是在哪里针对输入图像的resize调整了输出检测框的resize
         2. 检查在dataloader拿到lidar2img后，forward过程中具体在哪里使用，如何将路测与车端的image voxel对齐坐标系（重点检查mmdet3d/models/detectors/vicfuser_voxel/vimi.py中VIMI类的extract_feat函数）
