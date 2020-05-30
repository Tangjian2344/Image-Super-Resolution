# Image Super-Resolution

## Contents

[Classified_by_time](#Classified_by_time)

- [CVPR2020](#CVPR2020)

- [ICLR2020](#ICLR2020)

[Classified_by_network_structure](#Classified_by_network_structure)

- [Linear_Networks](#Linear_Networks)
    -  [Early_upsampling_designs](#Early_upsampling_designs)
    - [Late_upsampling_designs](#Late_upsampling_designs)
- [Residual_Networks](#Residual_Networks)
    -  [Sigle-stage_networks](#Sigle-stage_networks)
    - [Multi-stage_networks](#Multi-stage_networks)
- [Recursive_Networks](#Recursive_Networks)
- [Progressive_Reconstruction_Designs](#Progressive_Reconstruction_Designs)
- [Desely_Connected_Networks](#Desely_Connected_Networks)
- [Muti-branch_Designs](#Muti-branch_Designs)
- [Attention_Based_Networks](#Attention_Based_Networks)
- [Multiple_Degradation_Handing_Networks](#Multiple_Degradation_Handing_Networks)
- [GAN_Models](#GAN_Models)  



## Classified_by_time

### CVPR2020

- **[RFB-ESRGAN]** Perceptual Extreme Super Resolution Network with Receptive Field Block|[[pdf]](https://arxiv.org/pdf/2005.12597.pdf) 

  <details>   
      <summary>Summary</summary>   
      We proposed a super resolution network with receptive field block based on Enhanced SRGAN. We call our network RFB-ESRGAN.
  </details>

- **[DRN]** Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/2004.00448.pdf) [[code]](https://github.com/guoyongcs/DRN)

  <details>   
      <summary>Summary</summary>   
      We propose a dual regression scheme by introducing an additional constraint on LR data to reduce the space of the possible functions.
  </details>

- **[LFSSR-ATO]** Light Field Spatial Super-resolution via Deep Combinatorial Geometry Embedding and Structural Consistency Regularization|[[pdf]](https://arxiv.org/pdf/2004.02215.pdf) [[code]](https://github.com/jingjin25/LFSSR-ATO)

  <details>   
      <summary>Summary</summary>   
      In this paper, we propose a novel learningbased LF spatial SR framework.
  </details>

- **[SPSR]** Structure-Preserving Super Resolution with Gradient Guidance|[[pdf]](https://arxiv.org/pdf/2003.13081.pdf) [[code]](https://github.com/Maclory/SPSR)

  <details>   
      <summary>Summary</summary>   
       In this paper, we propose a structure-preserving super resolution method to alleviate the undesired structural distortions in the recovered images while maintaining the merits of GAN-based methods to generate perceptual-pleasant details.
  </details>

- **[USRNET]** Deep Unfolding Network for Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/2003.10428.pdf) [[code]](https://github.com/cszn/USRNet)

  <details>
      <summary>Summary</summary>
      This paper proposes an end-to-end trainable unfolding network which leverages both learning based methods and model-based methods,to  handle the SISR problem with different scale factors, blur kernels and noise levels.
  </details>

- **[PULSE]** PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models|[[pdf]](https://arxiv.org/pdf/2003.03808.pdf) 

  <details>
      <summary>Summary</summary>
      We propose an alternative formulation ofthe super-resolution problem based on creating realistic SR images that downscale correctly.
  </details>

- **[EventSR]** EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning|[[pdf]](https://arxiv.org/pdf/2003.07640.pdf) 

  <details>
      <summary>Summary</summary>
      We propose a novel end-to-end pipeline that reconstructs LR images from event streams, enhances the image qualities, and upsamples the enhanced images, called EventSR
  </details>

- **[UDVD]** Unified Dynamic Convolutional Network for Super-Resolution with Variational Degradations|[[pdf]](https://arxiv.org/pdf/2004.06965.pdf) 

  <details>   
      <summary>Summary</summary>   
       This paper proposes a unified network to accommodate the variations from interimage (cross-image variations) and intra-image (spatial variations),to train a single network for wide-ranging and variational degradations.
  </details>

- **[Zooming Slow-Mo]** Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time **Video** Super-Resolution|[[pdf]](https://arxiv.org/pdf/2002.11616.pdf) [[code]](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

  <details>   
      <summary>Summary</summary>   
       We propose a onestage space-time video super-resolution framework, which directly synthesizes an HR slow-motion video from an LFR, LR video.
  </details>

- **[STARnet]** Space-Time-Aware Multi-Resolution **Video** Enhancement|[[pdf]](https://arxiv.org/pdf/2003.13170.pdf) [[code]](https://github.com/alterzero/STARnet)

  <details>   
      <summary>Summary</summary>   
       Our proposed model called STARnet superresolves jointly in space and time, to increasing spatial resolution of video frames and simultaneously interpolating frames to increase the frame rate.
  </details>

- **[MZSR]** Meta-Transfer Learning for **Zero-Shot** Super-Resolution|[[pdf]](https://arxiv.org/pdf/2002.12213.pdf) [[code]](https://github.com/JWSoh/MZSR)

  <details>   
      <summary>Summary</summary>   
       We present Meta-Transfer Learning for Zero-Shot SuperResolution (MZSR), which leverages ZSSR, to adress two problem. One is that existing methods cannot exploit internal information within a specific image. Another is that they are applicable only to the specific condition of data that they are supervised.
  </details>

- **[CutBlur]** Rethinking **Data Augmentatione** for Image Super-resolution: A Comprehensive Analysis and a New Strategy|[[pdf]](https://arxiv.org/pdf/2004.00448.pdf) [[code]](https://github.com/clovaai/cutblur)

  <details>   
      <summary>Summary</summary>   
       We proposed a data augmentation method called CutBlur that cuts a low-resolution patch and pastes it to the corresponding high-resolution image region and vice versa.
  </details>

### ICLR2020

这些是正在审查的匿名论文/These are anonymous paper under review

- Neural Differential Equations for Single Image Super-Resolution|[[pdf]](https://openreview.net/pdf?id=gzcnMUReFv) [[code]](https://github.com/TevenLeScao/BasicSR)
- Depth-Recurrent Residual Connections for Super-Resolution of Real-Time Renderings|[[pdf]](https://openreview.net/pdf?id=H1gW93NKvH)
- **[PNEN]** PNEN: Pyramid Non-Local Enhanced Networks|[[pdf]](https://openreview.net/pdf?id=BJl7WyHFDS)
- **[SRGRN]** Global reasoning network for image super-resolution|[[pdf]](https://openreview.net/pdf?id=S1gE6TEYDB)
- **[SRDGAN]** SRDGAN: learning the noise prior for Super Resolution with Dual Generative Adversarial Networks|[[pdf]](https://openreview.net/pdf?id=HJlP_pEFPH)
- **[CRNET]** CRNet: Image Super-Resolution Using A Convolutional Sparse Coding Inspired Network|[[pdf]](https://openreview.net/pdf?id=rJgqjREtvS)
- Lossless single image super resolution from low_quality jpg images|[[pdf]](https://openreview.net/pdf?id=r1l0VCNKwB)
- Super-Resolution via Conditional Implicit Maximum Likelihood Estimation|[[pdf]](https://openreview.net/pdf?id=HklyMhCqYQ)
- **[HighRes-net]** HighRes-net: Multi-Frame Super-Resolution by Recursive Fusion|[[pdf]](https://openreview.net/pdf?id=HJxJ2h4tPr)
- Pixel Co-Occurence Based Loss Metrics for Super Resolution Texture Recovery|[[pdf]](https://openreview.net/pdf?id=rylrI1HtPr)  



## Classified_by_network_structure

### Linear_Networks

#### 	  Early_upsampling_designs

- **[SRCNN]** Image Super-Resolution Using Deep Convolutional Networks|[[pdf]](https://arxiv.org/pdf/1501.00092.pdf) [[code]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

  <details>
      <summary> Summary</summary>
      The pioneering work of using deep learning to solve the problem of image super-resolution reconstruction.
  </details>

- **[VDSR]** Accurate Image Super-Resolution Using Very Deep Convolutional Networks|[[pdf]](https://arxiv.org/pdf/1511.04587.pdf) [[code]](https://github.com/huangzehao/caffe-vdsr)

  <details>
        <summary> Summary</summary>
      We present a highly accurate single-image superresolution (SR) method which uses a very deep convolutional network inspired by VGG-net used for ImageNet classification.
    </details>

- **[DnCNN]** Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising|[[pdf]](https://arxiv.org/pdf/1608.03981.pdf) [[code]](https://github.com/cszn/DnCNN)

  <details>
      <summary> Summary</summary>
      In this paper, we take one step forward by investigating the construction of feed-forward denoising convolutional neural networks (DnCNNs) to embrace the progress in very deep architecture, learning algorithm, and regularization method into image denoising.
  </details>

- **[IrCNN]** Learning Deep CNN Denoiser Prior for Image Restoration|[[pdf]](https://arxiv.org/pdf/1704.03264.pdf) [[code]](https://github.com/cszn/ircnn)

  <details>
      <summary>Summary</summary>
      This paper aims to train a set of fast and effective CNN (convolutional neural network) denoisers and integrate them into model-based optimization method to solve other inverse problems.
  </details>
  
  #### Late_upsampling_designs

- **[FSRCNN]** Accelerating the Super-Resolution Convolutional Neural Network|[[pdf]](https://arxiv.org/pdf/1608.00367.pdf) [[code]](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)

- <details>
      <summary> Summary</summary>
      In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure for faster and better SR.
  </details>

- **[ESPCN]** Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network|[[pdf]](https://arxiv.org/pdf/1609.05158.pdf) [[code]]()

  <details>
      <summary> Summary</summary>
      We propose a novel CNNarchitecture where the feature maps are extracted in the LR space.
  </details>

### Residual_Networks  

#### 	  Sigle-stage_networks

- **[EDSR]** Enhanced Deep Residual Networks for Single Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1707.02921.pdf) [[code]](https://github.com/LimBee/NTIRE2017)

  <details>
      <summary> Summary</summary>
      In this paper, we develop an enhanced deep super-resolution network (EDSR) with performance exceeding those ofcurrent state-of-the-art SR methods.
  </details>

- **[CARN]** Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network|[[pdf]](https://arxiv.org/pdf/1803.08664.pdf) [[code]](https://github.com/nmhkahn/CARN-pytorch)

  <details>
      <summary> Summary</summary>
      We design an architecture that implements a cascading mechanism upon a residual network and present variant models of the proposed cascading residual network to further improve efficiency.
  </details>
  
  #### Multi-stage_networks

- **[FormResNet]** FormResNet: Formatted Residual Learning for Image Restoration|[[pdf]](https://jianbojiao.com/pdfs/cvprw.pdf) 

  <details>
      <summary> Summary</summary>
      Different from previous deep learning-based methods to directly learn the mapping from damaged images to clean images, We aim to learn the structured details and recovering the latent clean image together, from the shared information between the corrupted image and the latent image.
  </details>

- **[BTSRN]** Balance two-stage Residual Networks for Image Super-Resolution|[[pdf]](https://static.aminer.cn/upload/pdf/program/5a260bfb17c44a4ba8a1c725_0.pdf) [[code]](https://github.com/ychfan/sr_ntire2017)

  <details>
      <summary> Summary</summary>
      In this paper, balanced two-stage residual networks (BTSRN) are proposed for single image super-resolution.
  </details>

- **[REDNet]**

### Recursive_Networks

- **[DRCN]** Deeply-Recursive Convolutional Network for Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1511.04491.pdf) [[code]](https://www.vlfeat.org/matconvnet/)

  <details>
      <summary> Summary</summary>
      We propose an image super-resolution method (SR) using a deeply-recursive convolutional network (DRCN).
  </details>

- **[DRRN]** Image Super-Resolution via Deep Recursive Residual Network |[[pdf]](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) [[code]](https://github.com/tyshiwo/DRRN_CVPR17)

  <details>
      <summary> Summary</summary>
      This paper proposes a very deep CNN model (up to 52 convolutional layers) named Deep Recursive Residual Network (DRRN) that strives for deep yet concise networks.
  </details>

- **[MemNet]** MemNet: A Persistent Memory Network for Image Restoration|[[pdf]](https://arxiv.org/pdf/1708.02209.pdf) [[code]](https://github.com/tyshiwo/MemNet)

  <details>
      <summary> Summary</summary>
      We propose a very deep persistent memory network (MemNet) that introduces a memory block, consisting of a recursive unit and a gate unit, to explicitly mine persistent memory through an adaptive learning process.
  </details>

### Progressive_Reconstruction_Designs

- **[SCN]** Scale-wise Convolution for Image Restoration|[[pdf]](https://arxiv.org/pdf/1912.09028.pdf) [[code]](https://github.com/ychfan/scn)

  <details>
      <summary> Summary</summary>
      Inspired from spatial-wise convolution for shift-invariance, “scale-wise convolution” is proposed to convolve across multiple scales for scale-invariance.
  </details>

- **[LapSRN]** Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks|[[pdf]](https://arxiv.org/pdf/1710.01992.pdf) [[code]](http://vllab.ucmerced.edu/wlai24/LapSRN/)

  <details>
      <summary> Summary</summary>
      In this paper, we propose the deep Laplacian Pyramid Super-Resolution Network for fast and accurate image super-resolution. 
  </details>

### Desely_Connected_Networks

- **[RDN]** Residual Dense Network for Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1802.08797.pdf) [[code]](https://github.com/yulunzhang/RDN)

  <details>
      <summary> Summary</summary>
      We propose a novel residual dense network (RDN) to make full use of the hierarchical features from the original low-resolution (LR) images in image SR.
  </details>

- **[DBPN]** Deep Back-Projection Networks For Super-Resolution|[[pdf]](https://arxiv.org/pdf/1803.02735.pdf) [[code]](https://github.com/alterzero/DBPN-Pytorch)

  <details>
      <summary> Summary</summary>
      We propose Deep Back-Projection Networks (DBPN), that exploit iterative up- and downsampling layers, providing an error feedback mechanism for projection errors at each stage.
  </details>

### Muti-branch_Designs

- **[IDN]** |Fast and Accurate Single Image Super-Resolution via Information Distillation Network [[pdf]](https://arxiv.org/pdf/1803.09454.pdf) [[code]](https://github.com/Zheng222/IDN-Caffe)

  <details>
      <summary> Summary</summary>
      We propose a deep but compact convolutional network to directly reconstruct the high resolution image from the original low resolution image, to solve the challenges of computational complexity and memory consumption in practice.
  </details>

### Attention_Based_Networks

- **[SelNet]** A Deep Convolutional Neural Network with Selection Units for Super-Resolution|[[pdf]](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Choi_A_Deep_Convolutional_CVPR_2017_paper.pdf)

  <details>
      <summary> Summary</summary>
      Inspired by linear-mapping technique used in other super-resolution (SR) methods, we reinterpret ReLU into point-wise multiplication of an identity mapping and a switch, and finally present a novel nonlinear unit, called a selection unit (SU).
  </details>

- **[RCAN]** Image Super-Resolution Using Very Deep Residual Channel Attention Networks|[[pdf]](Image Super-Resolution Using Very Deep Residual Channel Attention Networks) [[code]](https://github.com/yulunzhang/RCAN)

  <details>
      <summary> Summary</summary>
      we propose the very deep residual channel attention networks (RCAN) to treated Discriminatively features across channel.
  </details>

- **[SRRAM]** RAM: Residual Attention Module for Single Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1811.12043v1.pdf)

  <details>
      <summary> Summary</summary>
      In this paper, we propose a new attention method, which is composed ofnew channelwise and spatial attention mechanisms optimized for SR and a new fused attention to combine them.
  </details>

- **[DRLN]** Densely Residual Laplacian Super-Resolution|[[pdf]](https://arxiv.org/pdf/1906.12021.pdf)

  <details>
      <summary> Summary</summary>
      We present a compact and accurate super-resolution algorithm namely, Densely Residual Laplacian Network (DRLN). The proposed network employs cascading residual on the residual structure to allow the flow of low-frequency information to focus on learning high and mid-level features. In addition, deep supervision is achieved via the densely concatenated residual blocks settings, which also helps in learning from high-level complex features.
  </details>

- **[SAN]** Second-order Attention Network for Single Image Super-Resolution|[[pdf]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf) [[code]](https://github.com/daitao/SAN)

  <details>
      <summary> Summary</summary>
      We propose a second-order attention network (SAN) for more powerful feature expression and feature correlation learning.
  </details>

### Multiple_Degradation_Handing_Networks

- **[ZSSR]** "Zero-Shot”  Super-Resolution using Deep Internal Learning|[[pdf]](https://arxiv.org/pdf/1712.06087.pdf) [[code]](https://github.com/assafshocher/ZSSR)

  <details>
      <summary> Summary</summary>
      In this paper we introduce “Zero-Shot” SR, which exploits the power ofDeep Learning, but does not rely on prior training. We exploit the internal recurrence of information inside a single image, and train a small image-specific CNN at test time, on examples extracted solely from the input image itself.
  </details>

- **[MZSR]** Meta-Transfer Learning for **Zero-Shot** Super-Resolution|[[pdf]](https://arxiv.org/pdf/2002.12213.pdf) [[code]](https://github.com/JWSoh/MZSR)

  <details>
      <summary> Summary</summary>
       We present Meta-Transfer Learning for Zero-Shot SuperResolution (MZSR), which leverages ZSSR, to adress two problem. One is that existing methods cannot exploit internal information within a specific image. Another is that they are applicable only to the specific condition of data that they are supervised.
  </details>

- **[SRMD]** Learning a Single Convolutional Super-Resolution Network for Multiple Degradations [[pdf]](https://arxiv.org/pdf/1712.06116.pdf) [[code]](https://github.com/cszn/SRMD)

  <details>
      <summary> Summary</summary>
      we propose a general framework with dimensionality stretching strategy that enables a single convolutional super-resolution network to take two key factors ofthe SISR degradation process, i.e., blur kernel and noise level, as input. Consequently, the super-resolver can handle multiple and even spatially variant degradations, which significantly improves the practicability.
  </details>

### GAN_Models

- **[SRGAN]** Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|[[pdf]](https://arxiv.org/pdf/1609.04802.pdf) [[code]](https://github.com/tensorlayer/srgan)

  <details>
      <summary> Summary</summary>
      In this paper, we present SRGAN, a generative adversarial network (GAN) for image superresolution (SR).
  </details>

- **[EnhanceNet]** EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis|[[pdf]](https://arxiv.org/pdf/1612.07919.pdf) [[code]](https://github.com/msmsajjadi/EnhanceNet-Code)

  <details>
      <summary> Summary</summary>
      We propose a novel application ofautomated texture synthesis in combination with a perceptual loss focusing on creating realistic textures rather than optimizing for a pixelaccurate reproduction ofground truth images during training. 
  </details>

- **[SRfeat]** SRFeat: Single Image Super-Resolution with Feature Discrimination|[[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf) [[code]](https://github.com/HyeongseokSon1/SRFeat)

  <details>
      <summary> Summary</summary>
      In this paper, we propose a novel GAN-based SISR method that overcomes the limitation of existing GAN-based and produces more realistic results by attaching an additional discriminator that works in the feature domain.
  </details>

- **[ESRGAN]** ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks|[[pdf]](https://arxiv.org/pdf/1809.00219.pdf) [[code]](https://github.com/xinntao/ESRGAN)

  <details>
      <summary> Summary</summary>
  To alleviate  the unpleasant artifacts in SRGAN and further enhance the visual quality, we thoroughly study three key components of SRGAN – network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN).
  </details>

  

  

