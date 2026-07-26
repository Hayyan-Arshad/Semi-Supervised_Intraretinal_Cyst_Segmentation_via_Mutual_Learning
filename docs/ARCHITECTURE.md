# SemiSAM System Architecture

## End-to-End Data and Learning Pipeline

SemiSAM couples a deployable convolutional segmentation network with a promptable foundation model during training. The architecture is organized into four stages: patient-wise cohort construction, dual-branch forward propagation, semi-supervised mutual optimization, and CNN-only evaluation.

### 1. Patient-Wise Cohort Construction

```mermaid
flowchart TB
    RAW["Raw OCT volumes<br/>B-scans and cyst/fluid masks"] --> PID["Group volumes by patient ID"]
    PID --> SPLIT{"Patient-wise split<br/>No patient overlap"}

    SPLIT --> TRAIN["Training patients"]
    SPLIT --> VAL["Validation patients"]
    SPLIT --> TEST["Test patients"]

    subgraph TRAIN_DATA["Training dataset preparation"]
        direction TB
        TRAIN --> TSLICES["Extract 2D OCT slices"]
        TSLICES --> TH5["Write one HDF5 file per slice<br/>image H x W | label H x W"]
        TH5 --> TLIST["train_slices.list"]
        TLIST --> LABEL_SPLIT{"Semi-supervised label allocation"}
        LABEL_SPLIT --> LABELED["Labeled pool D_l<br/>Masks exposed to the loss"]
        LABEL_SPLIT --> UNLABELED["Unlabeled pool D_u<br/>Masks withheld from the loss"]
    end

    subgraph VAL_DATA["Validation dataset preparation"]
        direction TB
        VAL --> VH5["Write patient HDF5 volumes<br/>image D x H x W | label D x H x W"]
        VH5 --> VLIST["val.list"]
    end

    subgraph TEST_DATA["Test dataset preparation"]
        direction TB
        TEST --> EH5["Write patient HDF5 volumes<br/>image D x H x W | label D x H x W"]
        EH5 --> ELIST["test.list"]
    end

    classDef source fill:#eef4ff,stroke:#315b96,stroke-width:2px,color:#101828;
    classDef process fill:#ffffff,stroke:#667085,stroke-width:2px,color:#101828;
    classDef labeled fill:#e7f6f1,stroke:#16866f,stroke-width:2px,color:#101828;
    classDef unlabeled fill:#fff0ed,stroke:#d65f4b,stroke-width:2px,color:#101828;
    class RAW,PID,SPLIT source;
    class TRAIN,VAL,TEST,TSLICES,TH5,TLIST,LABEL_SPLIT,VH5,VLIST,EH5,ELIST process;
    class LABELED labeled;
    class UNLABELED unlabeled;
```

*Figure 1. Patient-level partitioning is performed before slice extraction. This prevents correlated B-scans from the same patient appearing across training, validation, and test cohorts.*

### 2. Dual-Branch Forward Propagation

```mermaid
flowchart TB
    DL["Labeled pool D_l"] --> SAMPLER["Two-stream batch sampler"]
    DU["Unlabeled pool D_u"] --> SAMPLER
    SAMPLER --> MIXED["Mixed batch<br/>B_l labeled + B_u unlabeled"]
    MIXED --> LOAD["Load image and binary mask from HDF5"]
    LOAD --> NORM["OCT intensity normalization<br/>z-score | min-max | clipped z-score"]
    NORM --> AUG["Training augmentation<br/>random rotation + flip + resize"]
    AUG --> X["OCT tensor x<br/>B x 1 x H x W"]
    LOAD --> Y["Ground-truth tensor y_l<br/>B_l x 1 x H x W"]

    subgraph CNN_BRANCH["Branch A: deployable CNN"]
        direction TB
        X --> CNN["EfficientNet-B2 U-Net"]
        CNN --> ZC["CNN logits z_c<br/>B x 1 x H x W"]
        ZC --> PC["CNN probabilities<br/>sigmoid z_c"]
    end

    subgraph PROMPT_PIPELINE["CNN-guided prompt pipeline"]
        direction TB
        PC --> DETACH["Stop gradient"]
        DETACH --> THRESH["Threshold foreground probability"]
        THRESH --> EMPTY{"Foreground present?"}
        EMPTY -->|Yes| BOX["Tight bounding box + margin"]
        EMPTY -->|Yes| POINT["Sample positive foreground point"]
        EMPTY -->|No| FULLBOX["Full-image bounding box"]
        EMPTY -->|No| NEGPOINT["Negative center point"]
        BOX --> PROMPTS["Box + point prompts"]
        POINT --> PROMPTS
        FULLBOX --> PROMPTS
        NEGPOINT --> PROMPTS
    end

    subgraph SAM_INPUT["Prompt-model image preparation"]
        direction TB
        X --> RGB["Repeat OCT channel 1 to 3"]
        RGB --> RESIZE["Resize to SAM encoder resolution"]
        RESIZE --> PREP["SAM preprocessing"]
    end

    subgraph SAM_BRANCH["Branch B: promptable model"]
        direction TB
        PREP --> IMAGE_ENCODER["SAM / MedSAM ViT-B image encoder"]
        PROMPTS --> PROMPT_ENCODER["Prompt encoder"]
        IMAGE_ENCODER --> MASK_DECODER["Mask decoder"]
        PROMPT_ENCODER --> MASK_DECODER
        MASK_DECODER --> UPSAMPLE["Upsample to OCT resolution"]
        UPSAMPLE --> ZP["Prompt-model logits z_p<br/>B x 1 x H x W"]
    end

    classDef data fill:#eef4ff,stroke:#315b96,stroke-width:2px,color:#101828;
    classDef cnn fill:#e7f6f1,stroke:#16866f,stroke-width:2px,color:#101828;
    classDef prompt fill:#fff0ed,stroke:#d65f4b,stroke-width:2px,color:#101828;
    classDef process fill:#ffffff,stroke:#667085,stroke-width:2px,color:#101828;
    class DL,DU,SAMPLER,MIXED,LOAD,NORM,AUG,X,Y data;
    class CNN,ZC,PC cnn;
    class IMAGE_ENCODER,PROMPT_ENCODER,MASK_DECODER,UPSAMPLE,ZP prompt;
    class DETACH,THRESH,EMPTY,BOX,POINT,FULLBOX,NEGPOINT,PROMPTS,RGB,RESIZE,PREP process;
```

*Figure 2. The CNN processes every OCT B-scan and produces the mask used to derive a detached box-point prompt. The same image and generated prompt condition SAM or MedSAM, producing a second segmentation at the original OCT resolution.*

### 3. Semi-Supervised Mutual-Learning Objective

```mermaid
flowchart TB
    ZCL["CNN logits on labeled samples<br/>z_c_l"] --> CNN_SUP["CNN supervised loss<br/>Dice + binary cross-entropy"]
    YL1["Ground-truth masks y_l"] --> CNN_SUP

    ZPL["Prompt logits on labeled samples<br/>z_p_l"] --> PROMPT_SUP["Prompt supervised loss<br/>Dice + binary cross-entropy"]
    YL2["Ground-truth masks y_l"] --> PROMPT_SUP

    ZCA["CNN logits on all samples<br/>z_c"] --> C_TO_P["Dice(z_p, stopgrad(z_c))"]
    ZPA["Prompt logits on all samples<br/>z_p"] --> C_TO_P
    ZCA --> P_TO_C["Dice(z_c, stopgrad(z_p))"]
    ZPA --> P_TO_C

    CNN_SUP --> LSUP["L_sup = 0.5 * (L_cnn + L_prompt)"]
    PROMPT_SUP --> LSUP
    C_TO_P --> LMUTUAL["L_mutual = 0.5 * (L_c_to_p + L_p_to_c)"]
    P_TO_C --> LMUTUAL

    ITER["Training iteration t"] --> WARMUP{"Supervised warmup complete?"}
    WARMUP -->|No| ZERO["lambda(t) = 0"]
    WARMUP -->|Yes| RAMP["Sigmoid consistency ramp-up<br/>lambda(t) approaches lambda_max"]
    ZERO --> WEIGHT["Consistency weight lambda(t)"]
    RAMP --> WEIGHT

    LSUP --> TOTAL["L_total = L_sup + lambda(t) * L_mutual"]
    LMUTUAL --> TOTAL
    WEIGHT --> TOTAL
    TOTAL --> ADAM["Joint Adam optimization"]
    ADAM --> CNN_UPDATE["Update CNN parameters"]
    ADAM --> PROMPT_UPDATE["Update trainable SAM / MedSAM parameters"]
    CNN_UPDATE --> POLY_CNN["Polynomial CNN learning-rate decay"]
    PROMPT_UPDATE --> POLY_PROMPT["Polynomial prompt-model learning-rate decay"]

    classDef supervised fill:#e7f6f1,stroke:#16866f,stroke-width:2px,color:#101828;
    classDef mutual fill:#fff0ed,stroke:#d65f4b,stroke-width:2px,color:#101828;
    classDef schedule fill:#fff8e8,stroke:#b7791f,stroke-width:2px,color:#101828;
    classDef optimize fill:#f3effa,stroke:#7557a6,stroke-width:2px,color:#101828;
    class ZCL,YL1,ZPL,YL2,CNN_SUP,PROMPT_SUP,LSUP supervised;
    class ZCA,ZPA,C_TO_P,P_TO_C,LMUTUAL mutual;
    class ITER,WARMUP,ZERO,RAMP,WEIGHT schedule;
    class TOTAL,ADAM,CNN_UPDATE,PROMPT_UPDATE,POLY_CNN,POLY_PROMPT optimize;
```

*Figure 3. Ground-truth supervision is restricted to the labeled subset, whereas bidirectional consistency is evaluated over the complete mixed batch. Stop-gradient targets isolate the learning direction of each consistency term.*

### 4. Patient-Wise Validation and CNN-Only Inference

```mermaid
flowchart TB
    subgraph VALIDATION["Patient-wise validation during training"]
        direction LR
        VOLUME["Held-out validation OCT volume"] --> VNORM["Apply training intensity normalization"]
        VNORM --> VCNN["Current EfficientNet-B2 U-Net"]
        VCNN --> VPROB["Slice-wise foreground probabilities"]
        VPROB --> VMASK["Binary cyst masks"]
        VMASK --> METRICS["Volume Dice and segmentation metrics"]
        METRICS --> BEST{"Mean Dice improved?"}
        BEST -->|Yes| SAVE["Save cnn_best.pth"]
        BEST -->|No| KEEP["Keep existing best checkpoint"]
    end

    subgraph INFERENCE["CNN-only test and deployment"]
        direction LR
        TESTVOL["Unseen patient OCT volume"] --> TNORM["Apply training intensity normalization"]
        CHECKPOINT["cnn_best.pth"] --> TCNN["EfficientNet-B2 U-Net only"]
        TNORM --> TCNN
        TCNN --> TPROB["Cyst probability maps"]
        TPROB --> TMASK["Final binary segmentations"]
        TMASK --> OUTPUT["Patient-level predictions and metrics"]
    end

    SAVE --> CHECKPOINT

    classDef validation fill:#eef4ff,stroke:#315b96,stroke-width:2px,color:#101828;
    classDef checkpoint fill:#f3effa,stroke:#7557a6,stroke-width:2px,color:#101828;
    classDef deployment fill:#e7f6f1,stroke:#16866f,stroke-width:2px,color:#101828;
    class VOLUME,VNORM,VCNN,VPROB,VMASK,METRICS,BEST,KEEP validation;
    class SAVE,CHECKPOINT checkpoint;
    class TESTVOL,TNORM,TCNN,TPROB,TMASK,OUTPUT deployment;
```

*Figure 4. Model selection uses held-out patient volumes and CNN Dice performance. The promptable branch is a training-time collaborator only; deployment requires the normalized OCT input and the selected CNN checkpoint.*
