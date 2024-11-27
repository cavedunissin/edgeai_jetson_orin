# [邊緣AI - 使用 NVIDIA Jetson Orin Nano 開發具備深度學習、電腦視覺與生成式AI 功能的 ROS2 機器人]
CAVEDU 出版之 Jetson Orin 系列單板電腦書籍範例與相關資源

## 作者
* 曾吉弘博士，CAVEDU教育團隊技術總監、MIT CSAIL訪問學者、國立台灣科技大學資訊工程學系助理教授(兼任)、NVIDIA Jetson AI ambassador (白金)
* 郭俊廷，CAVEDU教育團隊資深講師、NVIDIA Jetson AI specialist
* 楊子賢，CAVEDU教育團隊資深講師、NVIDIA Jetson AI specialist

<img src=[./pics/ambassador_DavidTseng.png](https://github.com/cavedunissin/deeplearning_robot_jetson_nano/raw/main/pics/ambassador_DavidTseng.png) width="400" height="">

## 章節
### 第一章：AI、神經網路與邊緣裝置
    1.1 邊緣運算裝置
    1.2 單板電腦
    1.3 NVIDIA線上資源
    1.4 NVIDIA Jetson 家族
    1.5 Jetson Orin Nano 開發套件開箱
### 第二章：Jetson Orin Nano 單板電腦
    2.1 開機前設定
       2.1.1 下載映像檔
       2.1.2 燒錄映像檔到micro SD記憶卡
       2.1.3 使用SSD安裝開機系統
       2.1.4 硬體架設與開機設定
    2.2 Jetson Orin Nano開機！
       2.2.1 Wi-Fi 連線
       2.2.2 SSH遠端連線
       2.2.3 USB對接電腦與Jetson Orin Nano
       2.2.4 jtop系統管理員
       2.2.5 攝影機設定與測試
### 第三章：深度學習結合視覺辨識
    3.1 OpenCV深度學習介紹
       3.1.1 OpenCV介紹
       3.1.2 Jetson Orin Nano上的OpenCV
       3.1.3 拍攝單張照片
       3.1.4 讀取、編輯、展示圖像
       3.1.5 提取顏色
       3.1.6 RGB、BGR、HSV等常見顏色格式
       3.1.7 圖片疊合與抽色圖像
       3.1.8 加入文字
    3.2 NVIDIA深度學習視覺套件包 - Jetson Inference
       3.2.1 安裝jetson-inference函式庫
       3.2.2 圖像辨識
       3.2.3 物件偵測
       3.2.4 圖像分割
       3.2.5 姿態估計
       3.2.6 動作辨識
       3.2.7 背景移除
       3.2.8 距離估計
### 第四章：整合深度視覺 - 景深攝影機
    4.1 Intel RealSense景深攝影機
       4.1.1 在Jetson Orin Nano上安裝RealSense 套件
       4.1.2 在RealSense Viewer中檢視深度影像
       4.1.3 RealSense的Python範例
       4.1.4 使用RealSense D435辨識人臉與距離
    4.2 ZED景深攝影機
       4.2.1 硬體介紹
       4.2.2 環境設定
       4.2.3 範例
### 第五章：ROS2機器作業系統
    5.1 ROS/ROS2
       5.1.1 ROS
       5.1.2 ROS2
    5.2 NVIDIA Issac ROS
    5.3 安裝ROS2
    5.4 RK ROS2移動平台
       5.4.1 機器人系統架構
    5.5 ROS2基本節點
       5.5.1 導航
       5.5.2 地圖
       5.5.3  分段路徑規劃與影像串流
       5.5.4 光達節點
    5.6 AI節點
    5.7 進階應用
       5.7.1 距離偵測搭配ZED2 
       5.7.2  ArUco標記辨識與跟隨
       5.7.3  攝影機標定校正
### 第六章：邊緣裝置結合生成式AI
    6.1 淺談生成式AI
    6.2 NVIDIA Jetson Generative AI Lab
       6.2.1 文字生成
       6.3.2 文字與影像生成
       6.3.3 Vision Transformers
       6.3.4 機器人與具身 
       6.3.5 圖片生成  
       6.3.6  RAG &向量資料庫 - Jetson Copilot
       6.3.7  聲音
       6.3.8  Agent Studio
       
## 相關連結
* 開機用 SD 卡映像檔下載：
* CAVEDU 技術部落格相關文章：[https://blog.cavedu.com/?s=jetson](https://developer.nvidia.com/embedded/downloads)
* 購書：尚未出版
* 購買 Jetson Orin 系列：[https://robotkingdom.com.tw/product/nvidia-jetson-nano-developer-kit-set-b01/](https://robotkingdom.com.tw/?s=orin&post_type=product)
<img src=https://robotkingdom.com.tw/wp-content/uploads/2020/03/IMG_5024-scaled.jpg width="500" height="">

## 各章注意事項與安裝指令
* under construction...

## 各章註解
**第一章：AI、神經網路與邊緣裝置**
*	[註1-1] Bringing AI to the device: Edge AI chips come into their own:  https://www2.deloitte.com/cn/en/pages/technology-media-and-telecommunications/articles/tmt-predictions-2020-ai-chips.html
*	[註1-2] 何謂邊緣運算？https://blogs.nvidia.com.tw/2019/10/22/what-is-edge-computing/
*	[註1-3] Raspberry Pi 4: https://www.raspberrypi.org/products/raspberry-pi-4-model-b/specifications/
*	[註1-4] tinyML: https://www.tinyml.org/summit/
*	[註1-5] NVIDIA developer blog: https://developer.nvidia.com/blog/tag/jetson-nano/
*	[註1-6] NVIDIA developer forum: https://forums.developer.nvidia.com/ https://devtalk.nvidia.com/default/board/139/embedded-systems/1
*	[註1-7] NVIDIA開發者網站：https://developer.nvidia.com/
*	[註1-8]NVIDIA DLI: https://www.nvidia.com/zh-tw/deep-learning-ai/education/
*	[註1-9] Getting Started With AI On Jetson Nano: https://courses.nvidia.com/courses/course-v1:DLI+C-RX-02+V1/about
*	[註1-10] Getting Started with DeepStream for Video Analytics on Jetson Nano: https://courses.nvidia.com/courses/course-v1:DLI+C-IV-02+V1/info
*	[註1-11] Jetson 人工智慧認證： https://developer.nvidia.com/embedded/learn/jetson-ai-certification-programs
*	[註1-12] Jetson Nano專案彙整：https://developer.nvidia.com/embedded/community/jetson-projects
*	[註1-13] Jetson系列原廠頁面：https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/ ; https://developer.nvidia.com/embedded/jetson-modules
*	[註1-14] CAVEDU技術部落格Jetson相關文章：blog.cavedu.com/?s=jetson
*	[註1-15] Jetson TX: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-tx2/
*	[註1-16] MIT Racecar: https://racecar.mit.edu/
*	[註1-17] Jetson AGX Xavier: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-agx-xavier/
*	[註1-18] Volvo 選用 NVIDIA DRIVE 開發量產車款：https://www.nvidia.com/zh-tw/about-nvidia/press-releases/2019/volvo-chooses-nvidia-drive-to-develop-production-models/
*	[註1-19]Jetson nano: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/
*	[註1-20] Jetbot open source robot project: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetbot-ai-robot-kit/
*	[註1-21] Duckietown – Learning Autonomy: https://www.duckietown.org/ ; https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/duckietown/
*	[註1-22] Jetson NX: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-xavier-nx/
*	[註1-23] JetRacer: https://github.com/NVIDIA-AI-IOT/jetracer)
*	[註1-24] Jetson系列規格比較: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/ ; https://developer.nvidia.com/embedded/jetson-nano-dl-inference-benchmarks
*	[註1-25] Jetson系列之神經網路推論速度比較：https://developer.nvidia.com/embedded/jetson-benchmarks 
*	[註1-26] Jetson Nano與其他SBC效能比較：https://devblogs.nvidia.com/jetson-nano-ai-computing/
*	[註1-27] Jetson Nano快速上手頁面：https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

**第二章：Jetson Nano 單板電腦**
* p64 CSI camera 測試指令：
<code>gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e</code>

*   [註2-1] Jetson Nano主頁面：https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro
*	[註2-2] Jetson Nano映像檔載點：https://developer.nvidia.com/jetson-nano-sd-card-image-441
*	[註2-3] balenaEtcher: https://www.balena.io/etcher/ NVIDIA有提供各作業系統的英文教學
*	[註2-4] Win32 Disk Imager:https://sourceforge.net/projects/win32diskimager/
*	[註2-5] Jetson Nano各接頭介紹：https://developer.nvidia.com/zh-cn/embedded/learn/get-started-jetson-nano-devkit
*	[註2-6] 機器人王國Jetson Nano 4GB套件包：https://robotkingdom.com.tw/product/rk-nvidia-jetson-nano-developer-dlikit/
*	[註2-7] Ubuntu系統操作：https://www.ubuntu-tw.org/
*	[註2-8] NVIDIA Jetson 開發者專區：https://developer.nvidia.com/embedded-computing
*	[註2-9] NVIDIA Jetson支援論壇：https://devtalk.nvidia.com/default/board/139/embedded-systems/1
*	[註2-10] puTTY連線程式：https://www.putty.org/
*	[註2-11] MobaXterm連線程式：https://mobaxterm.mobatek.net/
*	[註2-12] Pi Camera 規格：https://www.raspberrypi.org/products/camera-module-v2/
*	[註2-13] Raspberry Pi攝影機測試：https://github.com/JetsonHacksNano/CSI-Camera
*	[註2-14] OpenCV官網：https://opencv.org/
*	[註2-15] OpenCV Wiki: https://zh.wikipedia.org/wiki/OpenCV
*	[註2-16] OpenCV Github: https://github.com/opencv/opencv
*	[註2-17] OpenCV Tutorials: https://docs.opencv.org/master/d9/df8/tutorial_root.html
*	[註2-18] OpenCV中文網站：http://www.opencv.org.cn/
*	[註2-19] HSV色彩空間：https://zh.wikipedia.org/zh-tw/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4
*	[註2-20] 色彩轉換器：https://www.peko-step.com/zhtw/tool/hsvrgb.html 
*	[註2-21] 圖片疊合：https://www.twblogs.net/a/5bb03a202b7177781a0fdf6d
*	[註2-22] cv2.putText語法參數說明：https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html

**第三章：深度學習結合視覺辨識**
*	[註3-1] NVIDIA官網介紹Deploying Deep Learning: https://developer.nvidia.com/embedded/twodaystoademo
*	[註3-2] TenrsorRT: https://developer.nvidia.com/zh-cn/tensorrt
*	[註3-3] jetson-inference github網站：https://github.com/dusty-nv/jetson-inference
*	[註3-4] jetson-inference建置流程：https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md
*	[註3-5] 影像辨識Jetson-inference: Classifying Images with ImageNet: https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md
*	[註3-6] GoogleNet: https://arxiv.org/pdf/1409.4842.pdf
*	[註3-7] 物件偵測Jetson-inference: Locating Objects with DetectNet：https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md 
*	[註3-8] MS COCO資料集：https://cocodataset.org/#home
*	[註3-9] 影像語意分割Jetson-inference: Semantic Segmentation with SegNet：https://github.com/dusty-nv/jetson-inference/blob/master/docs/segnet-console-2.md 
*	[註3-10] PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/

**第六章：JetBot深度視覺機器人**
*	[註6-1] 什麼是深度學習：https://blogs.nvidia.com/blog/2015/02/19/deep-learning-2/ 
*	[註6-2] AI、機器學習與深度學習的差異：https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/ 	
*	[註6-3] 深度學習訓練與推論 https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/
*	[註6-4] 障礙迴避專案資料夾：https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance
*	[註6-5] 障礙迴避執行影片：
*	[註6-6] pytorch：https://pytorch.org/
*	[註6-7] alexnet神經網路：http://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf
*	[註6-8] ImageNet影像資料集：http://www.image-net.org/
*	[註6-9] 道路跟隨專案資料夾：https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following
*	[註6-10]道路跟隨影片：https://youtu.be/8vN29tz4omg
*	[註6-11]ResNet-18神經網路：https://pytorch.org/hub/pytorch_vision_resnet/

**第七章：Intel RealSense 深度視覺攝影機**
*	P229 安裝RealSenseviewer套件網址部分有更改(更改教學如下):https://blog.cavedu.com/2021/07/12/realsense/
*	[註7-1] Intel RealSense: https://www.intelrealsense.com/ ; https://www.intel.com.tw/content/www/tw/zh/architecture-and-technology/realsense-overview.html
*	[註7-2] Intel RealSense D435景深攝影機：https://www.intelrealsense.com/depth-camera-d435/
*	[註7-3] JetsonHacksNano: https://github.com/JetsonHacksNano
*	[註7-4] Intel RealSense SDK 安裝須知: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md  (網址已更新)
*	[註7-5] RealSense Viewer: https://www.intelrealsense.com/sdk-2/
*	[註7-6] RealSense code example:  https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples 
*	[註7-7] Pyrealsense2: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
*	[註7-8] 顏色格式： http://docs.ros.org/en/kinetic/api/librealsense/html/namespacers.html
*	[註7-9] 假彩色：https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
*	[註7-10] Haar cascade classifier檔案下載： https://github.com/opencv/opencv/tree/master/data/haarcascades
*	[註7-11] OpenCV Cascade Classifier語法說明：https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

