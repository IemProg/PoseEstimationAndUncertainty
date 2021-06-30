# PoseEstimationAndUncertainty
A repo containing popular papers about pose estimation and uncertainty quantification

| Paper Title | Link | Uncertainty Estimation | Code |  | Comment |
|-|-|-|-|-|-|
| MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation |  | Yes | https://github.com/vita-epfl/monoloco |  | Laplacian Loss + Dropout |
| Monoloco++ | https://arxiv.org/pdf/2008.10913.pdf | Yes | https://github.com/vita-epfl/monoloco |  | Change in architecture + Spherical coordinates + task error |
| OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation | https://arxiv.org/pdf/2103.10180v1.pdf | No |  |  |  |
| Structured Aleatoric Uncertainty in Human Pose Estimation | https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Gundavarapu_Structured_Aleatoric_Uncertainty_in_Human_Pose_Estimation_CVPRW_2019_paper.pdf | Yes |  |  | Idea: Joins are approximated by multivariate Gaussian distr to use it as aleatoric uncertainty |
| Toward fast and accurate human pose estimation via soft-gated skip connections | https://arxiv.org/pdf/2002.11098v1.pdf | No | https://github.com/shivamsouravjha/Pose-estimators |  | Custum Network, state of the art on MPII and LSP dataset |
| Epipolar Transformers | https://arxiv.org/pdf/2005.04551v1.pdf | No | https://github.com/yihui-he/epipolar-transformers |  | Transfomer: to capture 3D features to improve 2D pose estimation, to avoid occlusion, and oblique viewing. Evaluated only on Human36M |
| RMPE: Regional Multi-Person Pose Estimation               -- 2D Pose Estimation -- | https://arxiv.org/pdf/1612.00137v5.pdf | No | https://github.com/MVIG-SJTU/AlphaPose |  | Very fast, multi-person tracking  (from RGB Image/Video), to create accurate tracking bounding boxes then 2D pose estimation |
| LCR-Net Localization-Classification-Regression | CVPR2017 |  |  |  |  |
| XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera | https://arxiv.org/pdf/1907.00837v2.pdf |  |  |  |  |
| Integral Human Pose Regression | ECCV2018 |  | https://github.com/JimmySuen/integral-human-pose |  |  |
| Fast Uncertainty Quantification for Deep Object Pose Estimation | https://arxiv.org/pdf/2011.07748.pdf |  |  |  | Idea: Spearman's rank correlation between pose error and UQ |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| Relative Pose Estimation: |  |  |  |  |  |
| Camera Distance-aware Top-Down approach for 3D Multi-person pose estimation | https://openaccess.thecvf.com/content_ICCV_2019/papers/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.pdf | No | https://github.com/mks0601/3DMPPE_ROOTNET_RELEASEhttps://github.com/mks0601/3DMPPE_ROOTNET_RELEASE |  | Multi-stage : DetectNet(MaskRCNN) + PoseNet + RootNet+ No dropout |
| DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild | https://arxiv.org/pdf/2008.09457v1.pdf | No | https://github.com/naver/dope |  | Train each expert seperately, then froze them and add distrilation loss (same accuracy) |
| HDNet: Human Depth Estimation for Multi-Person Camera-Space Localization | https://arxiv.org/abs/2007.08943 | No | https://github.com/jiahaoLjh/HumanDepth |  | Extract feature with backbone (FPN) then feed it to 1\ CNN for 2 Pose estimation  2\Where the depth is estimated by GNN |
| Lifting Transformer for 3D Human Pose Estimation in Video | https://arxiv.org/pdf/2103.14304v3.pdf | No |  |  |  |
| SMAP: Single-Shot Multi-Person: Absolute 3D Pose Estimation | ECCV 2020 | No | https://zju3dv.github.io/SMAP/ |  |  |
| Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation | CVPR2020 | No | https://github.com/fabbrimatteo/LoCO |  |  |
| Deep Network for the Integrated 3D Sensing of Multiple People in Natural Images | NIPS2018 | No |  |  |  |
| Self-Supervised Learning of 3D Human Pose using Multi-view Geometry | CVPR2019 | No |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| Datasets: |  |  |  |  |  |
| MPII | https://paperswithcode.com/dataset/mpii-human-pose |  |  |  | The benchmark dataset for single person 2D pose estimation. The images were collected from YouTube videos, covering daily human activities with complex poses and image appearances. There are about 25k images. In about 29k annotated poses are for training and another 7k are for testing. |
| Human36M |  |  |  |  | The largest 3D human pose benchmark. The dataset is captured in controlled environment. It consists of 3.6 millions of video frames. 11 subjects (5 females and 6 males) are captured from 4 camera viewpoints, performing 15 activities. |
| 3DPW | http://virtualhumans.mpi-inf.mpg.de/3DPW/ |  |  |  |  |
| DensePose | http://densepose.org/ |  |  |  |  |
| COCO |  |  |  |  | Requires “in the wild” multi-person detection and pose estimation in challenging, uncontrolled conditions. The COCO train, validation, and test sets, containing more than 200k images and 250k person instances labeled with keypoints. 150k instances of them are publicly available for training and validation. The COCO evaluation defines the object keypoint similarity (OKS) and uses the mean average precision (AP) over 10 OKS thresholds as main competition metric [1] |
| MuCo | http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/ |  |  |  | MuCo-3DHP, the first large scale training data set showing real images of sophisticated multi-person interactions and occlusions. We synthesize a large corpus of multi-person images by compositing images of individual people (with ground truth from mutli-view performance capture) |
| Single Persom Pose Estimation: |  |  |  |  |  |
| Integral Human Pose Regression | ECCV2018 |  |  |  |  |
| Learning Monocular 3D Human Pose Estimation from Multi-view Images | CVPR2018 |  |  |  |  |
| Synthetic Occlusion Augmentation with Volumetric Heatmaps for the                                                              2018 ECCV PoseTrack Challenge | https://arxiv.org/abs/1809.04987v3 |  |  |  |  |
| Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation | https://arxiv.org/pdf/1804.01110v1.pdf |  |  |  |  |
| 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning | CVPR2018 |  |  |  |  |
| End-to-end Recovery of Human Shape and Pose | https://arxiv.org/pdf/1712.06584.pdf |  |  |  |  |
| Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose | https://arxiv.org/pdf/1611.07828.pdf |  |  |  |  |
| 3D Human Pose Estimation with 2D Marginal Heatmaps | https://arxiv.org/pdf/1806.01484.pdf |  |  |  | Idea: Regularization If we want to encourage heatmaps to mimic the shape of a specific probability distribution, we can minimise the the Jensen-Shannon divergence (JSD) [11] from that particular distribution. |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| Multi-Person Pose Estimation |  |  |  |  |  |
| Learning 3D Human Pose from Structure and Motion | https://arxiv.org/abs/1711.09250 |  |  |  |  |
| Monocular 3D Pose and Shape Estimation of Multiple People in Natural Scenes | CVPR2018 |  |  |  |  |
| LCR-Net++: Multi-person 2D and 3D Pose Detection in Natural Images | https://arxiv.org/pdf/1803.00455.pdf |  |  |  |  |
| Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB | https://arxiv.org/pdf/1712.03453.pdf |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| Related Papers: |  |  |  |  |  |
| Analysis on the Dropout Effect in Convolutional Neural Networks | http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf |  |  |  |  |
| Deep Ensembles: A Loss Landscape Perspective | https://arxiv.org/pdf/1912.02757.pdf |  |  |  |  |
| Weight Uncertainty in Neural Networks | https://arxiv.org/pdf/1505.05424.pdf |  |  |  |  |
| What Uncertainties Do We Need in Bayesian DeepLearning for Computer Vision? | https://papers.nips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf |  |  |  |  |
| Towads robust methods for CV | https://arxiv.org/pdf/1906.01620.pdf |  |  |  |  |
| Lightweight Probabilistic Deep Networks | https://openaccess.thecvf.com/content_cvpr_2018/papers/Gast_Lightweight_Probabilistic_Deep_CVPR_2018_paper.pdf |  |  |  |  |
