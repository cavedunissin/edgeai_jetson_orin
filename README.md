# [邊緣AI - 使用 NVIDIA Jetson Orin Nano 開發具備深度學習、電腦視覺與生成式AI 功能的 ROS2 機器人]
CAVEDU 出版之 Jetson Orin 系列單板電腦書籍範例與相關資源

## 作者
* 曾吉弘博士，CAVEDU教育團隊技術總監、MIT CSAIL訪問學者、國立台灣科技大學資訊工程學系助理教授(兼任)、[NVIDIA DLI大使白金級](https://www.nvidia.com/en-us/training/instructor-directory/bio/?instructorId=0038Z00002pvnqVQAQ)
* 郭俊廷，CAVEDU教育團隊資深講師、NVIDIA Jetson AI Specialist
* 楊子賢，CAVEDU教育團隊資深講師、NVIDIA Jetson AI Specialist

<img src="https://github.com/cavedunissin/deeplearning_robot_jetson_nano/raw/main/pics/ambassador_DavidTseng.png" width="400" alt="Ambassador David Tseng">

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
### 第五章：ROS2機器人作業系統
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
       5.5.3 分段路徑規劃與影像串流
       5.5.4 光達節點
    5.6 AI節點
    5.7 進階應用
       5.7.1 距離偵測搭配ZED2 
       5.7.2 ArUco標記辨識與跟隨
       5.7.3 攝影機標定校正
### 第六章：邊緣裝置結合生成式AI
    6.1 淺談生成式AI
    6.2 NVIDIA Jetson Generative AI Lab
       6.2.1 文字生成
       6.3.2 文字與影像生成
       6.3.3 Vision Transformers
       6.3.4 機器人與具身 
       6.3.5 圖片生成  
       6.3.6 RAG &向量資料庫 - Jetson Copilot
       6.3.7 聲音
       6.3.8 Agent Studio
       
## 勘誤
* 第六章 Page 6-44 有兩個圖6-28，會於下一版修正(2025/03)

## 相關連結
* 開機用 SD 卡映像檔下載：https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/jp62-orin-nano-sd-card-image.zip 
* CAVEDU 技術部落格相關文章：https://blog.cavedu.com/?s=jetson
* 購書：https://www.books.com.tw/products/0011015829
* 購買 Jetson Orin 系列：https://robotkingdom.com.tw/?s=orin&post_type=product

<img src="https://robotkingdom.com.tw/wp-content/uploads/2023/04/NVIDIAJETSON-Orin-NANO-Developer-kit_1-scaled.jpg" width="400" alt="Jetson Orin Nano">

## 各章注意事項與安裝指令

* 請見 ch03 - ch06 資料夾下 md 檔

### 第三章：深度學習結合視覺辨識
* 提供 3.1 OpenCV深度學習介紹等 py 檔
* 3.2 Jetson Inference 相關範例於安裝完成之後即會存在於 ~/jetson-inference 路徑下，只提供執行語法與較長指令

### 第四章：整合深度視覺 - 景深攝影機
*

### 第五章：ROS2機器人作業系統


### 第六章：邊緣裝置結合生成式AI

## 各章註解
### 第一章：AI、神經網路與邊緣裝置
    1. Bringing AI to the device: Edge AI chips come into their own:  https://www2.deloitte.com/cn/en/pages/technology-media-and-telecommunications/articles/tmt-predictions-2020-ai-chips.html
    2. 何謂邊緣運算？https://blogs.nvidia.com.tw/2019/10/22/what-is-edge-computing/
    3. Raspberry Pi 4: https://www.raspberrypi.org/products/raspberry-pi-4-model-b/specifications/
    4. Raspberry Pi 5: https://www.raspberrypi.com/products/raspberry-pi-5/
    5. Edge AI Foundation(tinyML): https://www.tinyml.org
    6. NVIDIA 開發者部落格: https://developer.nvidia.com/blog/tag/jetson
    7. NVIDIA 開發者論壇: https://forums.developer.nvidia.com/
    8. NVIDIA DLI 深度學習機構: https://www.nvidia.com/en-us/training/ ; 
    9. NVIDIA開發者網站：https://developer.nvidia.com/
    10. Getting Started With AI on Jetson Nano: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2
    11. Jetson 人工智慧認證： https://developer.nvidia.com/embedded/learn/jetson-ai-certification-programs
    12. Jetson專案彙整：https://developer.nvidia.com/embedded/community/jetson-projects
    13. Jetson系列原廠頁面：https://developer.nvidia.com/embedded/jetson-modules
    14. CAVEDU技術部落格Jetson相關文章：https://blog.cavedu.com/?s=jetson
    15. Jetson TX: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-tx2/
    16. MIT Racecar: https://racecar.mit.edu/
    17. Jetson AGX Xavier: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-agx-xavier/
    18. NVIDIA 自駕車：https://www.nvidia.com/en-us/self-driving-cars/
    19. Jetson Nano: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/
    20. Jetbot 開放原始碼機器人專案： https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetbot-ai-robot-kit/
    21. Duckietown – Learning Autonomy: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/duckietown/
    22. Jetson NX: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-xavier-nx/
    23. JetRacer: https://github.com/NVIDIA-AI-IOT/jetracer
    24. Jetson系列規格比較: https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/
    25. Jetson benchmarks: https://developer.nvidia.com/embedded/jetson-benchmarks
    26. https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/
    27. https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html

### 第二章：Jetson Orin Nano 單板電腦
    1. https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit
    2. https://blog.cavedu.com/2022/08/02/nvidia-jetson-agx-orin-ssd-os/
    3. JetPack SDK: https://developer.nvidia.com/embedded/jetpack 
    4. balenaEtcher燒錄軟體：https://www.balena.io/etcher/
    5. Win32 Disk Imager燒錄軟體：https://sourceforge.net/projects/win32diskimager/
    6. https://robotkingdom.com.tw/product/nvidia-jetson-orin-nano-super-developer-set-1/
    7. NVIDIA SDK Manager: https://developer.nvidia.com/nvidia-sdk-manager
    8. 疏漏
    9. Ubuntu系統操作：https://www.ubuntu-tw.org/
    10. NVIDIA Jetson 開發者專區：https://developer.nvidia.com/embedded-computing
    11. NVIDIA開發者論壇/Jetson：https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70
    12. puTTY連線程式：https://www.putty.org/
    13. MobaXterm連線程式：https://mobaxterm.mobatek.net/
    14. jetson_stats: https://github.com/rbonghi/jetson_stats
    15. IMX219-160 Camera: https://www.waveshare.com/wiki/IMX219-160_Camera
    16. https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera
    17. Jetson Orin Nano Super：https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/
    18. https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/?ncid=so-face-299542
    19. https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/

### 第三章：深度學習結合視覺辨識
    1. OpenCV官網：https://opencv.org/
    2. OpenCV Wiki: https://zh.wikipedia.org/wiki/OpenCV
    3. OpenCV Github: https://github.com/opencv/opencv
    4. OpenCV Tutorials: https://docs.opencv.org/master/d9/df8/tutorial_root.html
    5. HSV色彩空間：https://w.wiki/Dmkc
    6. 色彩轉換器：https://www.peko-step.com/zhtw/tool/hsvrgb.html 
    7. cv2.putText語法參數說明：https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html
    8. NVIDIA 官網介紹 Deploying Deep Learning: https://developer.nvidia.com/embedded/twodaystoademo
    9. jetson-inference github網站：https://github.com/dusty-nv/jetson-inference
    10. TenrsorRT: https://developer.nvidia.com/zh-cn/tensorrt
    11. jetson-inference建置流程：https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md
    12. 圖像辨識Jetson-inference: Classifying Images with ImageNet: https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md
    13. GoogleNet: https://arxiv.org/pdf/1409.4842.pdf
    14. 物件偵測Jetson-inference: Locating Objects with DetectNet：https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md 
    15. MS COCO資料集：https://cocodataset.org/#home
    16. 圖像語意分割Jetson-inference: Semantic Segmentation with SegNet：https://github.com/dusty-nv/jetson-inference/blob/master/docs/segnet-console-2.md 
    17. PASCAL VOC資料集: http://host.robots.ox.ac.uk/pascal/VOC/
    18. 姿態估計:https://github.com/dusty-nv/jetson-inference/blob/master/docs/posenet.md
    19. 動作辨識:https://github.com/dusty-nv/jetson-inference/blob/master/docs/actionnet.md
    20. 背景移除:https://github.com/dusty-nv/jetson-inference/blob/master/docs/backgroundnet.md
    21. 距離估計:https://github.com/dusty-nv/jetson-inference/blob/master/docs/depthnet.md

### 第四章：整合深度視覺 - 景深攝影機
    1. Intel RealSense：https://www.intelrealsense.com/
       https://www.intel.com.tw/content/www/tw/zh/architecture-and-technology/realsense-overview.html
    2. Intel RealSense D435景深攝影機：https://www.intelrealsense.com/depth-camera-d435/
    3. JetsonHacksNano: https://github.com/jetsonhacks/installRealSenseSDK
    4. intelrealsense SDK: https://github.com/IntelRealSense/librealsense
    5. RealSense Viewer: https://github.com/IntelRealSense/librealsense/tree/master/tools/realsense-viewer
    6. RealSense 程式範例：https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples
    7. Pyrealsense2: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
    8. Intel RealSense opencv_viewer_example.py: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    9. 顏色格式：https://docs.ros.org/en/kinetic/api/librealsense/html/namespacers.html 
    10. OpenCV假彩色：https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
    11. Haar cascade classifier檔案下載： https://github.com/opencv/opencv/tree/master/data/haarcascades
    12. OpenCV Cascade Classifier語法說明：https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    13. ZED: https://www.stereolabs.com
    14. ZED2i 景深攝影機產品頁面: https://www.stereolabs.com/en-tw/products/zed-2
    15. ZED SDK: https://www.stereolabs.com/en-tw/developers/release
    16. ZED範例：https://www.stereolabs.com/docs/tutorials
    17. CAVEDU技術部落格ZED相關文章：https://blog.cavedu.com/?s=zed

### 第五章：ROS2機器作業系統
    1. RK ROS 機器人平台：https://robotkingdom.com.tw/?s=ROS2&post_type=product
    2. ROS https://www.ros.org/
    3. ROS2 https://github.com/ros2
    4. ROS版本清單 https://wiki.ros.org/Distributions
    5. NVIDIA Issac ROS: https://developer.nvidia.com/isaac/ros
    6. https://developer.nvidia.com/project-gr00t
    7. https://github.com/NVIDIA-ISAAC-ROS
    8. Navigation2: https://docs.nav2.org/
    9. ROS deep learning 節點清單：https://github.com/dusty-nv/ros_deep_learning
    10. ArUco標記辨識 ROS2套件 – aruco_ros: https://github.com/pal-robotics/aruco_ros
### 第六章：邊緣裝置結合生成式AI
    1. Jetson AI Lab:https://www.jetson-ai-lab.com/
    2. 模型效能比較：https://www.jetson-ai-lab.com/benchmarks.html
    3. 教學：https://www.jetson-ai-lab.com/tutorial-intro.html
    4. 文字生成：https://www.jetson-ai-lab.com/tutorial_text-generation.html
    5. jetson-containers: https://github.com/dusty-nv/jetson-containers
    6. ram-optimization: https://www.jetson-ai-lab.com/tips_ram-optimization.html
    7. live-llava: https://www.jetson-ai-lab.com/tutorial_live-llava.html
    8. VideoQuery: https://dusty-nv.github.io/NanoLLM/agents.html#video-query
    9. NanoDB: https://www.jetson-ai-lab.com/tutorial_nanodb.html
    10. nanoowl: https://github.com/NVIDIA-AI-IOT/nanoowl
    11. COCO 資料集: https://cocodataset.org/
    12. Track-Anything: https://github.com/gaomingqi/Track-Anything
    13. Meta SAM分割模型：https://github.com/facebookresearch/segment-anything
    14. Track-Anything:https://github.com/gaomingqi/Track-Anything/blob/master/doc/tutorials.md
    15. NVIDIA Cosmos世界基礎模型：https://www.nvidia.com/zh-tw/ai/cosmos/
    16. NVIDIA 推論微服務 (NVIDIA Inference Microservice)：https://blog.cavedu.com/2024/06/08/nvidia-nim-ai/
    17. Cosmos - World Foundation Models:https://www.jetson-ai-lab.com/cosmos.html
    18. LeRobot機器手臂專案：https://github.com/huggingface/lerobot/
    19. Action Chunking with Transformers: https://github.com/tonyzhaozh/act
    20. OpenVLA: https://openvla.github.io/
    21. Florence-2: https://huggingface.co/microsoft/Florence-2-large
    22. Open X-Embodiment 資料集：https://robotics-transformer-x.github.io/
    23. cuMotion動作規劃: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion
    24. Bing Image Creator: https://www.bing.com/images/create
    25. Adobe firefly: https://firefly.adobe.com/inspire/images
    26. Midjourney: https://www.midjourney.com
    27. Stable Diffusion: https://stability.ai/stable-image
    28. https://github.com/AUTOMATIC1111/stable-diffusion-webui
    29. Jetson Copilot: https://www.jetson-ai-lab.com/tutorial_jetson-copilot.html
    30. OpenAI Whisper :https://github.com/openai/whisper
    31. Agent Studio https://www.jetson-ai-lab.com/agent_studio.html

