<!-- # CategoryDiscovery-Survey -->

<p align="center">
  <h1 align="center">Category Discovery: An Open-World Perspective</h1>
      
  <p align="center">
    <a href="https://zhenqi-he.github.io/"><strong>Zhenqi He</strong></a>
    ,
    <a href="https://scholar.google.com/citations?user=GHTB15QAAAAJ&hl=en"><strong>Yuanpei Liu</strong></a>
    ,
    <a href="https://www.kaihan.org/"><strong>Kai Han</strong></a>
    

   </p>
   
</p>
<br />

This repository serves as a supplementary resource for our survey paper on Category Discovery (CD) methods. It includes a comprehensive collection of key papers, frameworks, and approaches in the field of CD, summarizing the most recent advancements and techniques. The materials here aim to provide researchers with an accessible overview of current trends and methodologies in CD, along with references and additional insights to support further exploration.

We will continue to maintain and update this repository with new papers and resources as the field evolves. Contributions are welcome, and we encourage pull requests (PRs) to help expand and improve the content for the community.


## Table of Contents

- [Introduction](#introduction)
- [Category Discovery](#category-discovery)
    - [Novel Category Discovery](#novel-category-discovery-ncd)
    - [Generalized Category Discovery](#generalized-category-discovery-gcd)
    - [Continual Category Discovery](#continual-category-discovery-ccd)
    - [On-the-fly Category Discovery](#on-the-fly-category-discovery-ocd)
    - [Category Discovery with domain shift](#category-discovery-with-domain-shift)
    - [Distribution-Agnostic Category Discovery](#distribution-agnostic-category-discovery-da-cd)
    - [Semantic Category Discovery](#semantic-category-discovery-scd)
    - [Few-Shot Category Discovery](#few-shots-category-discovery-fs-cd)
    - [Federated Category Discovery](#federated-category-discovery-fcd)


## Introduction
<!-- ![Alt Text](imgs/intro.jpg) -->
<div style="text-align:center;">
    <img src=./intro.png alt="Alt text" width="500"/>
</div>

Category Discovery (CD) addresses the limitations of the closed-world assumption by embracing an open-world setting.  As shown in above figure, CD differs from semi-supervised learning and OSR\&OOD by clustering unlabelled data that contains unseen categories.
It is motivated by that human beings are capable of discovering unknown species by transferring existing knowledge on explored species.
CD proves highly applicable across various real-world scenarios. For example, in autonomous driving, vehicles must continuously detect and classify new objects—such as unfamiliar road signs or obstacles—beyond their initial training to ensure safe navigation. In retail, CD can automatically recognize newly introduced products in supermarkets without the need for manual labeling.

### Roadmap
<!-- ![Alt Text](imgs/roadmap.jpg) -->
<div style="text-align:center;">
    <img src=./roadmap.jpg alt="Alt text" width="800"/>
</div>
In recent years, CD has garnered increasing attention, leading to a proliferation of research exploring various methodologies and settings. It was initially introduced as Novel Category Discovery (NCD) in to cluster unlabelled novel categories by leveraging knowledge from labelled base categories. This concept was later expanded into Generalized Category Discovery (GCD), which relaxed earlier constraints by assuming that the unlabelled data contains both novel and base categories, thereby more closely mirroring real-world scenarios. 
Further advancing the field, Han *etal.* proposed Semantic Category Discovery (SCD), aiming to assign semantic labels to unlabelled samples from an unconstrained vocabulary space.
Additionally, CD has been applied to complex scenarios such as continual learning, where models learn incrementally over time, and federated learning, which focuses on training models across decentralized devices while ensuring data privacy. 
CD methods have also been explored in challenging settings, including few-shot learning, where limited labelled data is available, and with imbalanced distribution and domain-shifted data, making CD more applicable to real-world problems.

## Category Discovery
<div style="text-align:center;">
    <img src=./taxonomies.png alt="Alt text" width="800"/>
</div>


### Novel Category Discovery (NCD)
The concept of NCD aims to transfer the knowledge learned from base categories to cluster unlabelled unseen categories, motivated by the observation where a child could easily distinguish novel categories (e.g., birds and elephants) after learning to classify base categories (e.g., dogs and cats).

Formally, given a dataset $\mathcal{D} = \mathcal{D}_L \cup \mathcal{D}_U$, where the labelled portion is $\mathcal{D}_L = \{(\mathbf{x}_i, y_i)\}_{i=1}^M \subset \mathcal{X} \times \mathcal{Y}_L$ and the unlabelled portion is $\mathcal{D}_U = \{(\mathbf{x}_i, \hat{y}_i)\}_{i=1}^K \subset \mathcal{X} \times \mathcal{Y}_U$ (with the labels $\hat{y}_i$ being inaccessible during training), the objective of NCD is to leverage the discriminative information learned from the annotated data to cluster the unlabelled data. 

This setting presumes that the label spaces of the labelled and unlabelled data are disjoint, i.e., $\mathcal{Y}_L \cap \mathcal{Y}_U = \varnothing$, implying $\mathcal{C}_N = \mathcal{Y}_U$, while also assuming a high degree of semantic similarity between the base and novel categories.




| Year   | Method                                     | Pub.        | Backbone        | Label Assignment         | # Unlabelled categories | Dataset                                                                                   |
|--------|--------------------------------------------|-------------|-----------------|--------------------------|-------------------------|-------------------------------------------------------------------------------------------|
| 2018   | [KCL](https://arxiv.org/abs/1711.10125)                                        | *ICLR*     | ResNet       | Parametric Classifier     | Over-estimate           | Omniglot, ImageNeg-1K, Office31                                                            |
| 2019   | [MCL](https://arxiv.org/abs/1901.00544)               | *ICLR*     | ResNet, VGG, LeNet | Parametric Classifier     | Over-estimate           | Omniglot, CIFAR-10&100, ImageNet-1K, MNIST                                                  |
|        | [DTC](https://arxiv.org/abs/1908.09884)                     | *ICCV*     | ResNet, VGG  | Soft Assignment           | $k$-Means               | Omniglot, CIFAR-10&100, ImageNet-1K, SVHN                                                   |
| 2020   | [RS, RS+](https://arxiv.org/abs/2002.05714)       | *ICLR*     | ResNet       | Parametric Classifier     | Known                   | Omniglot, CIFAR-10&100, ImageNet-1K, SVHN                                                   |
| 2021   | [Qing *etal.*](https://www.sciencedirect.com/science/article/abs/pii/S0893608021000575)           | *Neural Networks* | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, SVHN                                                                          |
|        | [OpenMix](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)               | *CVPR*     | ResNet, VGG  | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [NCL](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Neighborhood_Contrastive_Learning_for_Novel_Class_Discovery_CVPR_2021_paper.pdf)                            | *CVPR*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [JOINT](https://arxiv.org/abs/2104.12673)              | *ICCV*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [UNO](https://openaccess.thecvf.com/content/ICCV2021/papers/Fini_A_Unified_Objective_for_Novel_Class_Discovery_ICCV_2021_paper.pdf)                       | *ICCV*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [DualRS](https://arxiv.org/abs/2107.03358)                      | *ICCV*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100&1K, SSB                                                         |
| 2022   | [SMI](https://ieeexplore.ieee.org/document/9747827)                  | *ICASSP*   | VGG-16          | $k$-Means                 | Known                   | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [PSSCNNCD](https://ieeexplore.ieee.org/document/9409777)                  | *T'CYB*    | N/A             | BKBH $k$-Means            | Progressive label propagation | Coil20, Yeast, MSRA25, PalmData25, Abalone, USPS, Letter, MNIST                          |
|        | [Li *etal.*](https://arxiv.org/abs/2209.09120)| *NeurIPSW* | ResNet       | $k$-Means                 | $k$-Means               | CIFAR-100, ImageNet-1K                                                                     |
| 2023   | [ResTune](https://ieeexplore.ieee.org/document/9690577)                   | *T'NNLS*   | ResNet       | $k$-Means                 | Known                   | CIFAR-10&100, TinyImageNet                                                                  |
|        | [SK-Hurt](https://openreview.net/forum?id=oqOBTo5uWD&noteId=7UFcW08yOJ)      | *TMLR*     | ResNet       | $k$-Means                 | $k$-Means               | CIFAR-100, ImageNet-1K                                                                     |
|        | [IIC](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Modeling_Inter-Class_and_Intra-Class_Constraints_in_Novel_Class_Discovery_CVPR_2023_paper.pdf)               | *CVPR*     | ResNet       | Parametric Classifier     | $k$-Means               | CIFAR-10&100, ImageNet-1K                                                                   |
|        | [NSCL](https://arxiv.org/abs/2308.05017)                 | *ICML*     | ResNet       | $k$-Means                 | $k$-Means               | CIFAR-100, ImageNet-1K                                                                     |
|        | [CRKD](https://arxiv.org/abs/2307.09158)              | *ICCV*     | ResNet, ViT | Parametric Classifier   | Known                   | CIFAR-100, SSB                                                                               |
|        | [Feng *etal.*](https://arxiv.org/abs/2309.16451)         | *MICCAI*  | ResNet       | Parametric Classifier     | Known                   | ISIC2019                                                                                   |
| 2024   | [RAPL](https://arxiv.org/pdf/2405.06283)            | *CVPR*     | ResNet       | $k$-Means                 | Known                   | SoyAgeing                                                                                  |
|        | [SCKD](https://arxiv.org/abs/2407.01930) | *ECCV* | ResNet, ViT | Parametric Classifier | Known                   | CIFAR-10&100, ImageNet-100, SSB                                                             |
|        | [APL](https://arxiv.org/abs/2208.00979)                     | *T'PAMI*   | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, Omniglot, ImageNet-1K                                                          |
| PrePrint| [Hasan *etal.*](https://arxiv.org/abs/2307.03856) | *ArXiv* | ResNet | Parametric Classifier | $k$-Means               | CIFAR-10&100                                                                                 |

### Generalized Category Discovery (GCD)

Extending the NCD paradigm, Generalized Category Discovery relaxes the disjointness assumption between the base and novel categories, thereby presenting a more challenging and realistic scenario.
In GCD, the labelled and unlabelled datasets may share common categories, i.e., $\mathcal{Y}_L \cap \mathcal{Y}_U \neq \varnothing$, and the set of novel categories is defined as a subset of $\mathcal{Y}_U$ (i.e., $\mathcal{C}_N \subset \mathcal{Y}_U$). This general formulation is particularly pertinent to practical applications such as plant species discovery, where an existing database of known species is augmented with newly observed species, necessitating the clustering of both known and novel instances.

Notably, an equivalent formulation has been introduced by [Cao *etal.*](https://arxiv.org/abs/2102.03526) under the designation of Open-World Semi-Supervised Learning. In the following context, we refer to both formulations under the umbrella term Generalized Category Discovery.



| Year     | Method                                    | Pub.        | Backbone            | Label Assignment         | # Unlabelled categories | Dataset                                                                                       |
|----------|-------------------------------------------|-------------|---------------------|--------------------------|-------------------------|-----------------------------------------------------------------------------------------------|
| 2022     | [GCD](https://www.robots.ox.ac.uk/~vgg/research/gcd/)                | *CVPR*     | ViT            | Semi-$k$-Means           | $k$-Means               | CIFAR-10&100, ImageNet-100, SSB, Herb19                                                       |
|          | [ORCA](https://arxiv.org/abs/2102.03526)                      | *CVPR*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, Single-Cell                                                       |
|          | [ComEx](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Divide_and_Conquer_Compositional_Experts_for_Generalized_Novel_Class_Discovery_CVPR_2022_paper.pdf)               | *CVPR*     | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100                                                                                   |
|          | [OpenLDN](https://arxiv.org/abs/2207.02261)           | *ECCV*     | ResNet           | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, TinyImage, Oxford Pets                                            |
|          | [TRSSL](https://arxiv.org/abs/2207.02269)             | *ECCV*     | ResNet           | Parametric Classifier     | $k$-Means               | CIFAR-10&100, ImageNet-100, TinyImage, Oxford Pets, Scars, Aircrafts                          |
|          | [NACH](https://proceedings.neurips.cc/paper_files/paper/2022/file/15dce910311b9bd82ca24f634148519a-Paper-Conference.pdf)                   | *NeurIPS*  | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100                                                                     |
|          | [XCon](https://arxiv.org/abs/2208.01898)                  | *BMVC*     | ViT            | Semi-$k$-Means           | $k$-Means               | CIFAR-10&100, ImageNet-100, SSB, Oxford Pets                                                  |
| 2023     | [OpenCon](https://arxiv.org/abs/2208.02764)             | *TMLR*     | ResNet       | Prototype-based          | $k$-Means               | CIFAR-10&100, ImageNet-100                                                                     |
|          | [PromptCAL](https://arxiv.org/abs/2212.05590)       | *CVPR*     | ViT            | Semi-$k$-Means           | Known                   | CIFAR-10&100, ImageNet-100, SSB                                                               |                                          |
|          | [DCCL](https://arxiv.org/abs/2303.17393)                 | *CVPR*     | ViT            | Infomap                 | Infomap                 | CIFAR-10&100, ImageNet-100, CUB, Scars, Oxford Pets                                           |
|          | [OpenNCD](https://arxiv.org/abs/2305.13095)              | *IJCAI*    | ResNet           | Prototype-based          | Prototype Grouping      | CIFAR-10&100, ImageNet-100                                                                     |
|          | [SimGCD](https://arxiv.org/abs/2211.11727)               | *ICCV*     | ViT            | Parametric Classifier     | $k$-Means               | CIFAR-10&100, ImageNet-100, SSB, Herb19                                                       |
|          | [GPC](https://arxiv.org/abs/2305.06144)              | *ICCV*     | ViT            | GMM                      | GMM                     | CIFAR-10&100, ImageNet-100, SSB                                                               |
|          | [PIM](https://arxiv.org/abs/2212.00334)         | *ICCV*     | ViT            | Parametric Classifier     | $k$-Means               | CIFAR-10&100, ImageNet-100, CUB, Scars, Herb19                                                |
|          | [TIDA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c646b713f5de2cf1ab1939d49a4036d-Abstract-Conference.html)              | *NeurIPS*  | ResNet       | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, TinyImageNet, Scars, Aircraft                                      |
|          | [$\mu$GCD](https://arxiv.org/abs/2311.17055)            | *NeurIPS*  | ResNet, ViT, ViT | $k$-Means             | Known                   | Clevr-4                                                                                         |
|          | [InfoSieve](https://arxiv.org/abs/2310.19776)       | *NeurIPS*  | ViT            | $k$-Means                 | $k$-Means               | CIFAR-10&100, ImageNet-100, SSB, Oxford Pets, Herb19                                         |
|          | [SORL](https://arxiv.org/abs/2311.03524)                 | *NeurIPS*  | ResNet           | $k$-Means                 | Known                   | CIFAR-10&100                                                                                   |
|          | [Yang *etal.*](https://arxiv.org/abs/2310.19210) | *ICONIP*  | ViT            | Louvain                  | Louvain                 | CIFAR-10&100, ImageNet-100, CUB, Scars, Herb19                                                 |
| 2024     | [AMEND](https://openaccess.thecvf.com/content/WACV2024/papers/Banerjee_AMEND_Adaptive_Margin_and_Expanded_Neighborhood_for_Efficient_Generalized_Category_WACV_2024_paper.pdf)           | *WACV*     | ViT            | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, SSB, Herb19                                                       |
|          | [GCA](https://openaccess.thecvf.com/content/WACV2024/papers/Otholt_Guided_Cluster_Aggregation_A_Hierarchical_Approach_to_Generalized_Category_Discovery_WACV_2024_paper.pdf)               | *WACV*     | ViT            | Guided Cluster Aggregation| $k$-Means               | CIFAR-10&100, ImageNet-100, SSB                                                               |
|          | [SPT-Net](https://arxiv.org/abs/2403.13684)             | *ICLR*     | ViT, ViT  | Parametric Classifier     | $k$-Means               | CIFAR-10&100, ImageNet-100, SSB                                                               |
|          | [LegoGCD](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_Solving_the_Catastrophic_Forgetting_Problem_in_Generalized_Category_Discovery_CVPR_2024_paper.pdf)             | *CVPR*     | ViT            | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, -1K, SSB, Herb19                                                  |
|          | [CMS](https://arxiv.org/abs/2404.09451)            | *CVPR*     | ViT            | Agglomerative Clustering  | Agglomerative Clustering | CIFAR-100, ImageNet-100, SSB, Herb19                                                          |
|          | [ActiveGCD](https://arxiv.org/abs/2403.04272)             | *CVPR*     | ViT            | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, SSB                                                               |
|          | [TextGCD](https://arxiv.org/abs/2403.07369) | *ECCV* | ViT | Parametric Classifier | Known | CIFAR-10&100, ImageNet-100, -1K, SSB, Oxford Pets, Flowers102                                 |
|          | [LPS](https://arxiv.org/abs/2309.11930)                  | *IJCAI*    | ResNet      | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100                                                                     |                                               |
|          | [Contextuality-GCD](https://arxiv.org/abs/2407.19752) | *ICIP* | ViT | Parametric Classifier | Known | CIFAR-10&100, ImageNet-100, -1K, SSB, Herb19                                                   |
| 2025     | [MSGCD](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000934)           | *Information Fusion*     | ViT            | Parametric Classifier     | Known                   | CIFAR-100, SSB    
|      | [CPT](https://link.springer.com/article/10.1007/s11263-024-02343-w)           | *IJCV*     | ViT            | Similarity-Based     | $k$-Means                 | CIFAR-10&100, ImageNet-100,CUB, Scars, Herb19           
|      | [PAL-GCD](https://github.com/Terminator8758/PAL-GCD)           | *AAAI*     | ViT            | Parametric Classifier   | DBSCAN               | CIFAR-100,ImageNet-100,SSB, Herb19     
|      | [DebGCD](https://openreview.net/forum?id=9B8o9AxSyb)           | *ICLR*     | ViT            | Parametric Classifier   | DBSCAN               | CIFAR-10&100,ImageNet-100&1K,SSB, Herb19,Oxford-Pets   
|      | [ProtoGCD](https://openreview.net/forum?id=9B8o9AxSyb)           | *T'PAMI*     | ViT            | Parametric Classifier   | $k$-Means                | CIFAR-10&100,ImageNet-100&1K,SSB, Herb19   
|      | [MOS](https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_MOS_Modeling_Object-Scene_Associations_in_Generalized_Category_Discovery_CVPR_2025_paper.pdf)           | *CVPR*     | ViT            | Parametric Classifier   | Known                | SSB, Oxford-Pets  
|      | [GET](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_GET_Unlocking_the_Multi-modal_Potential_of_CLIP_for_Generalized_Category_CVPR_2025_paper.pdf)           | *CVPR*     | ViT            | Parametric Classifier   | Known               | CIFAR-10&100,ImageNet-100,SSB,Herb19
|      | [AptGCD](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Less_Attention_is_More_Prompt_Transformer_for_Generalized_Category_Discovery_CVPR_2025_paper.pdf)           | *CVPR*     | ViT            | Parametric Classifier   | Known               | CIFAR-10&100,ImageNet-100,SSB,Herb19
|      | [Dai *et al*](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Less_Attention_is_More_Prompt_Transformer_for_Generalized_Category_Discovery_CVPR_2025_paper.pdf)           | *CVPR*     | ViT            | -   | Known               | SSB, Herb19
|      | [HypCD](https://visual-ai.github.io/hypcd/)           | *CVPR*     | ViT            | -   | Known               | CIFAR-10&100,ImageNet-100,SSB,Herb19
| PrePrint | [CLIP-GCD](https://arxiv.org/abs/2305.10420) | *ArXiv* | ViT  | Semi-$k$-Means           | $k$-Means               | CIFAR-10&100, ImageNet-100, -1K, SSB, Flowers102, DomainNet                                  |
|          | [MCDL](https://arxiv.org/abs/2401.13325) | *ArXiv* | ViT         | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, -1K, CUB, SCars, Herb19                                           |
|          | [PNP](https://arxiv.org/abs/2404.08995)   | *ArXiv*    | ViT            | Infomap                 | Infomap                 | CIFAR-10&100, ImageNet-100, -1K, SSB, Herb19                                                  |
|          | [RPIM](https://arxiv.org/abs/2405.20711) | *ArXiv* | ViT  | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, -1K, CUB, Scars, Herb19                                           |
|          | [OpenGCD](https://arxiv.org/abs/2308.06926) | *ArXiv*  | ViT  | Parametric Classifier     | $k$-Means               | CIFAR-10&100, CUB                                                                                |
|          | [ConceptGCD](https://arxiv.org/abs/2410.13285)      | *ArXiv*    | ViT, ViT  | Parametric Classifier     | $k$-Means               | CIFAR-100, ImageNet-100, -1K, SSB, Herb19                                                     |
|          | [GET](https://arxiv.org/abs/2403.09974) | *ArXiv* | ViT          | Parametric Classifier     | Known                   | CIFAR-10&100, ImageNet-100, SSB, Herb19                                                       |


### Continual Category Discovery (CCD)

CCD provides a continual setting of category discovery in which new categories are identified sequentially while retaining previously acquired knowledge.
CCD presents several distinct scenarios based on the structure of the incoming data. 
In the **Class Incremental Scenario*, the training set $\mathcal{D}_{train}^t$ contains solely unlabelled instances from novel categories. In the *Mixed Incremental Scenario*, $\mathcal{D}_{train}^t$ is composed exclusively of unlabelled data drawn from both novel and base categories. Finally, in the *Semi-Supervised Mixed Incremental Scenario*, $\mathcal{D}_{train}^t$ comprises both labelled and unlabelled samples, which originate from the base as well as the novel categories.

| Year   | Method                                                                 | Pub.      | Backbone               | Scenario                | Label Assignment         | # Unlabelled categories | Dataset                                      |
|--------|------------------------------------------------------------------------|-----------|------------------------|-------------------------|--------------------------|-------------------------|----------------------------------------------|
| 2022   | [NCDwF](https://arxiv.org/abs/2207.10659)                  | *ECCV*   | ResNet              | Class Incremental       | Parametric Classifier     | Known                   | CIFAR-10/100, ImageNet-1K                   |
|        | [FRoST](https://arxiv.org/abs/2207.08605)                                              | *ECCV*   | ResNet              | Class Incremental       | Parametric Classifier     | Known                   | CIFAR-10/100, TinyImageNet                 |
|        | [GM](https://arxiv.org/abs/2210.04174)                         | *NeurIPS*| ResNet              | All                     | Parametric Classifier     | Known                   | CIFAR-100, ImageNet-100, CUB               |
| 2023   | [PA-GCD](https://arxiv.org/abs/2307.10943)                                        | *ICCV*   | ViT, Resnet| Mixed Incremental        | Parametric Classifier     | Affinity Propagation     | CUB, MIT67, Stanford Dogs, Aircraft        |
|        | [MetaGCD](https://arxiv.org/abs/2308.11063)                                         | *ICCV*   | ViT               | Mixed Incremental        | $k$-Means                | $k$-Means               | CIFAR-10/100, TinyImageNet                 |
|        | [iGCD](https://arxiv.org/abs/2304.14310)                                      | *ICCV*   | ResNet              | Self-Supervised Mixed Incremental | Soft Nearest Neighbor   | Density Peaks           | CUB, Aircraft, CIFAR-100                   |
| 2024   | [Msc-iNCD](https://arxiv.org/abs/2303.15975)                                          | *ICPR*   | ViT               | Class Incremental       | Parametric Classifier     | Known                   | CIFAR-100, ImageNet-100/1K                 |
|        | [ADM](https://arxiv.org/abs/2403.03382)               | *AAAI*   | ResNet              | Class Incremental       | Parametric Classifier     | Known                   | CIFAR-10/100, TinyImageNet                 |
|        | [PromptCCD](https://arxiv.org/abs/2407.19001)                                   | *ECCV*   | ViT               | Mixed Incremental        | GMM                      | GMP                      | CIFAR-100, ImageNet-100, TinyImageNet       |
|        | [DEAN](https://arxiv.org/abs/2408.13492)                | *ECCV*   | ViT               | Mixed Incremental        | Parametric Classifier     | Affinity Propagation     | CUB, Aircraft, CIFAR-100                   |
|        | [CAMP](https://arxiv.org/abs/2308.12112)                                        | *ECCV*   | ViT             | Self-Supervised Mixed Incremental | Nearest Centroid Classifier | Known                   | CUB, Aircraft, SCars, DomainNet, CIFAR-100 |
|        | [Happy](https://arxiv.org/abs/2410.06535)                                             | *NeurIPS*| ViT               | Mixed Incremental        | Parametric Classifier     | Silhouette Score         | CIFAR-100, ImageNet-100, TinyImageNet, CUB |
| Preprint| [FEA](https://arxiv.org/abs/2405.06389)                         | *ArXiv*  | ViT               | Class Incremental       | Parametric Classifier     | Known                   | CIFAR-10/100, TinyImageNet                 |

### On-the-fly Category Discovery (OCD)

OCD extends conventional category discovery to an inductive learning paradigm with streaming inference. It trains on a labelled support set $D_S$ to cluster unlabelled query set $D_Q$ where $D_S$ is unavailable during training and its samples are individually at test time.

| Year     | Method                                    | Pub.        | Backbone            | Label Assignment         | # Unlabelled categories | Dataset                                                                                       |
|----------|-------------------------------------------|-------------|---------------------|--------------------------|-------------------------|-----------------------------------------------------------------------------------------------|
|    2023      | [SMILE](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_On-the-Fly_Category_Discovery_CVPR_2023_paper.pdf)                    | *CVPR*     | ViT            | Hash-based              | Hash-coding             | CIFAR-10&100, ImageNet-100, CUB, Scars, Herb19                                                | 
|     2024     | [PHE](https://arxiv.org/abs/2410.19213)          | *NeurIPS*  | ViT            | Hamming Ball-Based       | Hamming Ball-Based       | CUB, Scars, Oxford Pets, Food-101, iNaturalist   

### Category Discovery with domain shift

This setting relaxes the conventional assumption that both labelled and unlabelled data are drawn from the same semantic domain. Formally, let $\mathcal{D}_L$ denote the labelled data, assumed to be exclusively drawn from the domain $\Omega_ and let $\mathcal{D}_U$ denote the unlabelled data, which may include samples originating from both $\Omega_and an additional domain $\Omega_{N}$. The objective is to accurately classify images drawn from the combined domain $\Omega = \Omega_cup \Omega_{N}$, under the assumption that the novel domain is disjoint from the base domain (i.e., $\Omega_cap \Omega_{N} = \varnothing$). In practice, the novel domain $\Omega_{N}$ may encompass multiple subdomains.

| Year   | Method                                                                 | Pub.      | Backbone  | $ \Omega_{\mathcal{U}} $     | Label Assignment     | # Unlabelled categories | Dataset                                            | $ \mathcal{Y_L} \cap \mathcal{Y_U} $ |
|--------|------------------------------------------------------------------------|-----------|-----------|--------------------------------|----------------------|-------------------------|----------------------------------------------------|------------------------------------------|
| 2022   | [Yu *etal.*](https://ojs.aaai.org/index.php/AAAI/article/view/20224)                              | *AAAI*   | ResNet | Single New Domain             | Parametric Classifier | $k$-Means               | Office, OfficeHome, VisDA                         | $ \varnothing $                       |
|        | [SCDA](https://arxiv.org/abs/2203.03329)                         | *ICME*   | ResNet | Multiple New Domains          | Parametric Classifier | $k$-Means               | Office, OfficeHome, DomainNet                     | $ \varnothing $                       |
| 2023   | [SAN](https://arxiv.org/abs/2211.11262)                     | *ICCV*   | ResNet | Single New Domain             | Parametric Classifier | N/A                     | Office, OfficeHome, VisDA, DomainNet              | $ \varnothing $                       |
| 2024   | [CDAD-Net](https://arxiv.org/abs/2404.05366)                   | *CVPRW*  | ViT  | Single New Domain             | Semi-$k$-Means        | Elbow                   | OfficeHome, PACS, DomainNet, CIFAR-10&100, ImageNet-100 | $ \neq \varnothing $                  |
| 2025| [HiLo](https://arxiv.org/abs/2408.04591)                   | *ICLR*  | ViT  | Multiple new Domains          | Parametric Classifier | $k$-Means               | DomainNet, SSB-C                                   | $ \neq \varnothing $                  |
| ArXiv       | [Wang *etal.*](https://arxiv.org/abs/2406.18140)                 | *ArXiv*  | ViT  | Single New Domain             | Parametric Classifier | Known                   | CIFAR-10, OfficeHome, DomainNet                   | $ \varnothing $                       |


### Distribution-Agnostic Category Discovery (DA-CD)

DA-CD eliminates the requirement for a balanced distribution imposed on both labelled and unlabelled data in conventional category discovery. Instead, it acknowledges that the data may follow a skewed distribution, such that for certain categories $\mathcal{Y}_i$ and $\mathcal{Y}_j$ within the set $\mathcal{Y}$ it holds that
$\mathbb{P}_{\mathcal{Y_x}}(\mathcal{Y}_i) > \mathbb{P}_{\mathcal{Y_x}}(\mathcal{Y}_j).$
In this formulation, the set $\mathcal{Y_x}$ may refer to either the labelled categories $\mathcal{Y}_L$ or the unlabelled categories $\mathcal{Y}_U$.

| Year   | Method                                                                 | Pub.     | Backbone        | Scenario                              | Label Assignment     | # Unlabelled categories | Dataset                                                |
|--------|------------------------------------------------------------------------|----------|-----------------|---------------------------------------|----------------------|-------------------------|--------------------------------------------------------|
| 2023   | [NCDLR](https://openreview.net/forum?id=ey5b7kODvK)                                             | *TMLR*  | ViT        | Long-tailed Distribution for $ \mathcal{Y_L} \& \mathcal{Y_U} $ | Parametric Classifier | $k$-Means               | CIFAR-10, ImageNet-100, Herb19, iNaturalist18          |
|        | [ImbaGCD](https://arxiv.org/abs/2401.05353)              | *CVPRW* | Resnet   | Imbalanced Distribution for $ \mathcal{Y_U} $ | Parametric Classifier | Known                   | CIFAR-10\&100, ImageNet-100                             |
|        | [GCDLR](https://arxiv.org/abs/2401.05352)            | *ICCVW* | Resnet   | Imbalanced Distribution for $ \mathcal{Y_U} $ | Parametric Classifier | Known                   | CIFAR-10\&100, ImageNet-100                             |
|        | [BYOP](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Bootstrap_Your_Own_Prior_Towards_Distribution-Agnostic_Novel_Class_Discovery_CVPR_2023_paper.html)                                          | *CVPR*  | ResNet       | Imbalanced Distribution for $ \mathcal{Y_U} $ | Parametric Classifier | Known                   | CIFAR-10\&100, TinyImageNet                             |
|        | [BaCon](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html)                                            | *NeurIPS* | ViT       | Long-tailed Distribution for $ \mathcal{Y_L} \& \mathcal{Y_U} $ | $k$-Means             | Known                   | CIFAR-10\&100-LT, ImageNet-100-LT, Places-LT            |
| 2024   | [Fan *etal*](https://arxiv.org/abs/2403.01053)                                          | *CVPR*  | ViT        | Long-tailed Distribution for $ \mathcal{Y_L} \& \mathcal{Y_U} $ | $k$-Means             | Spectral graph           | BioMedical Datasets                                     |


### Semantic Category Discovery (SCD)

In contrast to NCD and GCD, which focus solely on grouping visually similar images without considering their semantic meaning, SCD extends these paradigms by also assigning a semantic label to each unlabelled instance. Specifically, SCD leverages an open vocabulary label space to achieve this goal. In this context, WordNet, comprising approximately 68,000 labels, is employed as a comprehensive and unconstrained vocabulary, facilitating the assignment of meaningful semantic labels.

| Year  | Method                                                                 | Pub.      | Backbone            | Word Space                          | Label Assignment        | # Unlabelled categories | Dataset                                      |
|-------|------------------------------------------------------------------------|-----------|---------------------|--------------------------------------|-------------------------|-------------------------|----------------------------------------------|
| 2024  | [SCD](https://arxiv.org/abs/2304.02364)                                  | *CVPRW*  | ViT            | ~Open                           | KMeans+Top-$k$ Voting    | Known                   | ImageNet-100&1K, SCars, CUB                 |
|       | [SNCD](https://ojs.aaai.org/index.php/AAAI/article/view/28371)                                     | *AAAI*   | ResNet           | $\mathcal{C}_{base} + \mathcal{C}_{novel}$ | Parametric Classifier    | Known                   | CIFAR-10&100, ImageNet-100                  |

### Few-Shots Category Discovery (FS-CD)

FS-CD addresses the challenge of identifying novel classes when only a very limited amount of labelled data is available. This setting extends traditional category discovery by integrating the principles of few-shot learning. In particular, FSCD adopts an $N$-way, $k$-shot framework in which the model is required to discriminate among $N$ distinct classes with merely $k$ labelled examples per class for base categories.

[Chi *etal.*](https://arxiv.org/abs/2102.04002) extend NCD to a few-shot setting by linking it to meta-learning, based on the shared assumption that base and novel categories possess high-level semantic features. By adapting meta-learning techniques such as Model-Agnostic Meta-Learning and Prototypical Networks (ProtoNet), their approach shifts the focus from classification to clustering tasks—a critical adjustment for few-shot category discovery. A key innovation is the introduction of the Clustering-rule-aware Task Sampler, which ensures that training tasks adhere to consistent clustering rules, thereby enabling the model to generalize better to novel categories despite the limited labelled data. However, this method assumes that the number of novel categories is known in advance.

### Federated Category Discovery (FCD)

FCD extends Category Discovery in a federated learning setting, facilitating decentralized and collaborative model training among clients while safeguarding data privacy.

| Year   | Method                                                                 | Pub.     | Backbone        | Label Assignment         | # Unlabelled categories | Dataset                                       | 
|--------|------------------------------------------------------------------------|----------|-----------------|--------------------------|-------------------------|-----------------------------------------------|
| 2023   | [FedoSSL](https://arxiv.org/abs/2305.00771)             | *ICML*   | ResNet       | Parametric Classifier     | Known                   | CIFAR-10/100, CINIC-10                        |
| 2024   | [FedGCD](https://openaccess.thecvf.com/content/CVPR2024/papers/Pu_Federated_Generalized_Category_Discovery_CVPR_2024_paper.pdf)              | *CVPR*   | ViT        | GMM                      | Semi-FINCH              | CIFAR-10/100, ImageNet-100, CUB, SCars, Pets  |
| Preprint| [GAL](https://arxiv.org/abs/2312.13500)                        | *ArXiv*  | ResNet\&34   | Parametric Classifier     | Potential Prototype Merge | CIFAR-100, TinyImageNet, ImageNet-100         |
