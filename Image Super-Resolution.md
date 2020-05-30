# Image Super-Resolution

## Contents

[Classified-by-time](#Classified-by-time)

[	CVPR2020](#CVPR2020)

​	[ICLR2020](#ICLR2020)

[Classified by network structure](#Classified by network structure)

## Classified-by-time

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
       We present Meta-Transfer Learning for Zero-Shot SuperResolution (MZSR), which leverages ZSSR, to adress two problem. One problem is that existing methods cannot exploit internal information within a specific image. Another is that they are applicable only to the specific condition of data that they are supervised.
  </details>

- **[CutBlur]** Rethinking **Data Augmentatione** for Image Super-resolution: A Comprehensive Analysis and a New Strategy|[[pdf]](https://arxiv.org/pdf/2004.00448.pdf) [[code]](https://github.com/clovaai/cutblur)

  <details>   
      <summary>Summary</summary>   
       We proposed a data augmentation method called CutBlur that cuts a low-resolution patch and pastes it to the corresponding high-resolution image region and vice versa.
  </details>

### ICLR2020

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



## Classified by network structure

### Linear Networks

#### 	Early upsampling designs

- **[SRCNN]** Image Super-Resolution Using Deep Convolutional Networks|[[pdf]](https://arxiv.org/pdf/1501.00092.pdf) [[code]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

- **[VDSR]** Accurate Image Super-Resolution Using Very Deep Convolutional Networks|[[pdf]](https://arxiv.org/pdf/1511.04587.pdf) [[code]](https://github.com/huangzehao/caffe-vdsr)
- **[DnCNN]** Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising|[[pdf]](https://arxiv.org/pdf/1608.03981.pdf) [[code]](https://github.com/cszn/DnCNN)
- **[IrCNN]** Learning Deep CNN Denoiser Prior for Image Restoration|[[pdf]](https://arxiv.org/pdf/1704.03264.pdf) [[code]](https://github.com/cszn/ircnn)

#### 	Late upsampling designs

- **[FSRCNN]** Accelerating the Super-Resolution Convolutional Neural Network|[[pdf]](https://arxiv.org/pdf/1608.00367.pdf) [[code]](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
- **[ESPCN]** Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network|[[pdf]](https://arxiv.org/pdf/1609.05158.pdf) [[code]]()

### Residual Networks

#### 	Sigle-stage networks

- **[EDSR]** Enhanced Deep Residual Networks for Single Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1707.02921.pdf) [[code]](https://github.com/LimBee/NTIRE2017)
- **[CARN]** Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network|[[pdf]](https://arxiv.org/pdf/1803.08664.pdf) [[code]](https://github.com/nmhkahn/CARN-pytorch)

#### 	Multi-stage networks

- **[FormResNet]** FormResNet: Formatted Residual Learning for Image Restoration|[[pdf]](https://jianbojiao.com/pdfs/cvprw.pdf) 
- **[BTSRN]** Balance two-stage Residual Networks for Image Super-Resolution|[[pdf]](https://static.aminer.cn/upload/pdf/program/5a260bfb17c44a4ba8a1c725_0.pdf) [[code]](https://github.com/ychfan/sr_ntire2017)
- **[REDNet]**

### Recursive Networks

- **[DRCN]** Deeply-Recursive Convolutional Network for Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1511.04491.pdf) [[code]](https://www.vlfeat.org/matconvnet/)
- **[DRRN]** Image Super-Resolution via Deep Recursive Residual Network pdf|[[pdf]](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) [[code]](https://github.com/tyshiwo/DRRN_CVPR17)
- **[MemNet]** MemNet: A Persistent Memory Network for Image Restoration|[[pdf]](https://arxiv.org/pdf/1708.02209.pdf) [[code]](https://github.com/tyshiwo/MemNet)

### Progressive Reconstruction Designs

- **[SCN]** Scale-wise Convolution for Image Restoration|[[pdf]](https://arxiv.org/pdf/1912.09028.pdf) [[code]](https://github.com/ychfan/scn)
- **[LapSRN]** Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks|[[pdf]](https://arxiv.org/pdf/1710.01992.pdf) [[code]](http://vllab.ucmerced.edu/wlai24/LapSRN/)

### Desely Connected Networks

- **[RDN]** Residual Dense Network for Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1802.08797.pdf) [[code]](https://github.com/yulunzhang/RDN)
- **[D-DBPN]** Deep Back-Projection Networks For Super-Resolution|[[pdf]](https://arxiv.org/pdf/1803.02735.pdf) [[code]]https://github.com/alterzero/DBPN-Pytorch

### Muti-branch Designs

- **[IDN]** |Fast and Accurate Single Image Super-Resolution via Information Distillation Network [[pdf]](https://arxiv.org/pdf/1803.09454.pdf) [[code]](https://github.com/Zheng222/IDN-Caffe)

### Attention Based Networks

- **[SelNet]** A Deep Convolutional Neural Network with Selection Units for Super-Resolution|[[pdf]](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Choi_A_Deep_Convolutional_CVPR_2017_paper.pdf)
- **[RCAN]** Image Super-Resolution Using Very Deep Residual Channel Attention Networks|[[pdf]](Image Super-Resolution Using Very Deep Residual Channel Attention Networks) [[code]](https://github.com/yulunzhang/RCAN)
- **[SRRAM]** RAM: Residual Attention Module for Single Image Super-Resolution|[[pdf]](https://arxiv.org/pdf/1811.12043v1.pdf)
- **[DRLN]** Densely Residual Laplacian Super-Resolution|[[pdf]](https://arxiv.org/pdf/1906.12021.pdf)
- **[SAN]** Second-order Attention Network for Single Image Super-Resolution|[[pdf]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf) [[code]](https://github.com/daitao/SAN)

### Multiple Degradation Handing Networks

- **[ZSSR]** "Zero-Shot”  Super-Resolution using Deep Internal Learning|[[pdf]](https://arxiv.org/pdf/1712.06087.pdf) [[code]](https://github.com/assafshocher/ZSSR)
- **[MZSR]** Meta-Transfer Learning for **Zero-Shot** Super-Resolution|[[pdf]](https://arxiv.org/pdf/2002.12213.pdf) [[code]](https://github.com/JWSoh/MZSR)
- **[SRMD]** Learning a Single Convolutional Super-Resolution Network for Multiple Degradations [[pdf]](https://arxiv.org/pdf/1712.06116.pdf) [[code]](https://github.com/cszn/SRMD)

### GAN Models

- **[SRGAN]** Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|[[pdf]](https://arxiv.org/pdf/1609.04802.pdf) [[code]](https://github.com/tensorlayer/srgan)
- **[EnhanceNet]** EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis|[[pdf]](https://arxiv.org/pdf/1612.07919.pdf) [[code]](https://github.com/msmsajjadi/EnhanceNet-Code)
- **[SRfeat]** SRFeat: Single Image Super-Resolution with Feature Discrimination|[[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf) [[code]](https://github.com/HyeongseokSon1/SRFeat)
- **[ESRGAN] **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks|[[pdf]](https://arxiv.org/pdf/1809.00219.pdf) [[code]](https://github.com/xinntao/ESRGAN)

