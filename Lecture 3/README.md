# Lecture 3 — R-CNN Family: From Region Proposals to Instance Segmentation

Hands-on implementations of the **R-CNN family** of models in Jupyter notebooks. This lecture traces how object detection evolved from slow, multi-stage pipelines to unified, end-to-end architectures — and how **Mask R-CNN** extended detection to pixel-level instance segmentation.

## Notebooks

| Notebook | Model | What it implements |
|----------|-------|-------------------|
| `R_CNN_implementation.ipynb` | **R-CNN** | Selective Search proposals + VGG16 backbone + per-region classification and bounding-box regression |
| `Faster_RCNN_code.ipynb` | **Faster R-CNN** | Fine-tuning `torchvision` Faster R-CNN on the Aquarium COCO detection dataset |
| `Mask_RCNN.ipynb` | **Mask R-CNN** | Training `maskrcnn_resnet50_fpn` on Penn-Fudan Pedestrian for instance segmentation |

> **Note:** There is no separate Fast R-CNN notebook. Fast R-CNN is covered below as a key step in the architectural evolution between R-CNN and Faster R-CNN.

## Run

These notebooks were built for **Google Colab** (GPU recommended). Open a notebook in Colab or locally in Jupyter/VS Code with a CUDA-capable environment.

**Typical setup:**

```powershell
pip install torch torchvision opencv-python matplotlib albumentations pycocotools selectivesearch torch_snippets
```

| Notebook | Dataset | Notes |
|----------|---------|-------|
| `R_CNN_implementation.ipynb` | Custom CSV + images (Google Drive) | Mount Drive and point paths to your data folder |
| `Faster_RCNN_code.ipynb` | [Aquarium COCO](https://public.roboflow.com/object-detection/aquarium/2/download/coco) | Mount Drive; dataset path in notebook |
| `Mask_RCNN.ipynb` | [Penn-Fudan Pedestrian](https://www.cis.upenn.edu/~jshi/ped_html/) | Auto-downloads inside the notebook |

---

## Evolution of the R-CNN Family

```text
R-CNN (2014)          →  Slow: CNN run separately on ~2000 regions per image
Fast R-CNN (2015)     →  Shared convolutional features; single forward pass per image
Faster R-CNN (2015)   →  Region Proposal Network (RPN) replaces Selective Search
Mask R-CNN (2017)     →  Adds a parallel mask branch for instance segmentation
```

| Model | Region proposals | Feature extraction | Training | Inference speed |
|-------|------------------|------------------|----------|-----------------|
| R-CNN | External (Selective Search) | Per-region CNN forward pass | Multi-stage | Very slow |
| Fast R-CNN | External (Selective Search) | Once per image (shared) | Single-stage, multi-task loss | Slow |
| Faster R-CNN | Learned (RPN) | Once per image (shared) | End-to-end | Fast |
| Mask R-CNN | Learned (RPN) | Once per image (shared) | End-to-end + mask head | Fast |

---

## R-CNN (Regions with CNN Features)

**Paper:** Girshick et al., 2014 — *Rich feature hierarchies for accurate object detection and semantic segmentation*

### High-level pipeline

```text
Input image
    ↓
Selective Search  →  ~2000 region proposals (bounding boxes)
    ↓
For EACH proposal:
    Resize/warp to fixed size (e.g. 224×224)
    ↓
    CNN (e.g. VGG16)  →  fixed-length feature vector (4096-d)
    ↓
    SVM classifier  →  object class score
    ↓
    BBox regressor  →  refined box coordinates
    ↓
Non-Maximum Suppression (NMS)  →  final detections
```

### Architectural details

1. **Region proposals (Selective Search)**  
   A classical, non-learned algorithm that groups pixels by color/texture and produces category-agnostic box candidates. No gradient flows through this step.

2. **CNN backbone (VGG16)**  
   Each warped crop is passed independently through a pretrained CNN. Features are extracted **per region**, not shared across the image — this is the main bottleneck.

3. **Two task-specific heads (per region)**  
   - **Classification:** SVM (originally) or softmax layer — “what object is in this box?”  
   - **Bounding-box regression:** Predicts \((\Delta x, \Delta y, \Delta w, \Delta h)\) to refine proposal coordinates.

4. **NMS**  
   Overlapping high-scoring boxes are suppressed so one object is not detected multiple times.

### Limitations

- **Extremely slow** — thousands of forward passes per image  
- **Multi-stage training** — CNN fine-tuning, then SVMs, then bbox regressors trained separately  
- **Large disk/memory** — storing features for all proposals during training  

### This repo’s implementation (`R_CNN_implementation.ipynb`)

- Uses **Selective Search** via `selectivesearch`  
- **VGG16** backbone (classifier removed; features only)  
- Custom `RCNN` module with classification + regression heads  
- Trains on a custom detection dataset from Google Drive  

---

## Fast R-CNN

**Paper:** Girshick, 2015 — *Fast R-CNN*

Fast R-CNN fixes R-CNN’s biggest flaw: **redundant convolution**.

### High-level pipeline

```text
Input image
    ↓
Single CNN forward pass  →  convolutional feature map (shared for whole image)
    ↓
Selective Search proposals  →  projected onto feature map
    ↓
RoI Pooling  →  fixed-size feature per region (e.g. 7×7)
    ↓
Fully connected layers
    ↓
    ├─ Softmax  →  class + background
    └─ BBox regressor  →  refined coordinates
```

### Architectural details

1. **Shared feature extraction**  
   The entire image passes through the CNN **once**. All proposals read from the same feature map.

2. **RoI (Region of Interest) Pooling**  
   Each proposal is mapped to coordinates on the feature map. RoI Pooling divides the region into a fixed grid (e.g. 7×7) and max-pools each cell — producing a fixed-size tensor regardless of proposal size.

3. **Multi-task loss (single network)**  
   One loss combines classification and bounding-box regression:
   \[
   L = L_{\text{cls}} + \lambda \, L_{\text{box}}
   \]
   Training is end-to-end for the CNN + heads (proposals still external).

4. **No SVM**  
   Replaced by a softmax layer trained jointly with the regressor.

### Improvements over R-CNN

- ~10× faster training, ~100× faster inference at test time  
- Single-stage training (no separate SVM training)  
- Still depends on **external** Selective Search — proposals are not learned  

---

## Faster R-CNN

**Paper:** Ren et al., 2015 — *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*

Faster R-CNN replaces Selective Search with a **learned, differentiable** proposal generator.

### High-level pipeline

```text
Input image
    ↓
Backbone CNN (e.g. ResNet, MobileNet)  →  feature map
    ↓
┌─────────────────────────────────────┐
│  Region Proposal Network (RPN)      │
│  Sliding anchors on feature map     │
│  → objectness score + box deltas    │
│  → ~300 proposals                   │
└─────────────────────────────────────┘
    ↓
RoI Pooling / RoI Align  →  fixed features per proposal
    ↓
Detection head (Fast R-CNN head)
    ├─ Class prediction
    └─ BBox refinement
    ↓
NMS  →  final detections
```

### Architectural details

1. **Backbone + FPN (Feature Pyramid Network)**  
   Modern implementations (including `torchvision`) often use a multi-scale FPN built on top of the backbone. Small and large objects are detected at different feature-map resolutions.

2. **Region Proposal Network (RPN)**  
   - Places **anchor boxes** of multiple scales/aspect ratios at each spatial location on the feature map  
   - Two sibling 1×1 conv heads per anchor:  
     - **Objectness:** binary (object vs. background)  
     - **Box regression:** refine anchor to proposal  
   - Top-scoring proposals pass to the detection head  

3. **RoI Align (Mask R-CNN improvement; often used in Faster R-CNN too)**  
   Replaces RoI Pooling with bilinear interpolation — avoids quantization misalignment from snapping region boundaries to the discrete grid.

4. **End-to-end training**  
   RPN and detection head share the backbone. Losses from both stages are combined:
   \[
   L = L_{\text{RPN}} + L_{\text{det}}
   \]

5. **Near real-time inference**  
   Proposal generation is a single small network forward pass — no Selective Search.

### This repo’s implementation (`Faster_RCNN_code.ipynb`)

- Uses `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`  
- Fine-tunes on the **Aquarium** COCO-format dataset  
- Albumentations for augmentation; custom `AquariumDetection` dataset class  

---

## Mask R-CNN

**Paper:** He et al., 2017 — *Mask R-CNN*

Mask R-CNN extends Faster R-CNN with a **parallel mask prediction branch** — detecting objects **and** segmenting each instance at the pixel level.

### High-level pipeline

```text
Input image
    ↓
Backbone + FPN  →  feature maps
    ↓
RPN  →  region proposals
    ↓
RoI Align  →  fixed feature per proposal
    ↓
┌──────────────────────────────────────────┐
│  RoI Heads (parallel branches)           │
│  ├─ Class + box head  →  label + bbox     │
│  └─ Mask head         →  K×K mask / class │
└──────────────────────────────────────────┘
    ↓
NMS on detections; masks for surviving boxes
```

### Architectural details

1. **Same two-stage detector as Faster R-CNN**  
   Backbone, RPN, and bounding-box/classification head are unchanged in spirit.

2. **Mask branch (key addition)**  
   - A small FCN (fully convolutional network) runs on each RoI  
   - Outputs a **K×K** binary mask (e.g. 28×28) per region **per class**  
   - At inference, only the mask channel for the **predicted class** is used  
   - Masks are predicted in parallel with class/box — not sequentially  

3. **RoI Align (critical for masks)**  
   Pixel-accurate alignment matters more for segmentation than for boxes. RoI Align removes the harsh quantization of RoI Pooling.

4. **Multi-task loss**
   \[
   L = L_{\text{cls}} + L_{\text{box}} + L_{\text{mask}}
   \]
   - \(L_{\text{mask}}\): per-pixel sigmoid cross-entropy on the true class mask  

5. **Instance vs. semantic segmentation**  
   - **Semantic:** one label per pixel (all people = one class)  
   - **Instance (Mask R-CNN):** separate mask for **each object instance**  

### This repo’s implementation (`Mask_RCNN.ipynb`)

- Uses `maskrcnn_resnet50_fpn` from `torchvision`  
- Trains on **Penn-Fudan Pedestrian** (pedestrian instance masks)  
- Custom `PennFudanDataset`; replaces box and mask predictors for 2 classes (background + pedestrian)  
- Saves checkpoint as `maskrcnn_pennfudan.pth`  

---

## How the four models relate (summary diagram)

```text
                    ┌─────────────────────────────────────┐
                    │         Input Image                 │
                    └─────────────────┬───────────────────┘
                                      │
         R-CNN / Fast R-CNN           │           Faster R-CNN / Mask R-CNN
         (external proposals)         │           (learned RPN)
                    │                 │                 │
                    ▼                 ▼                 ▼
              Selective Search   Shared CNN       Backbone + FPN
                    │                 │                 │
                    │                 │                 ▼
                    │                 │              RPN (anchors)
                    │                 │                 │
                    └────────► RoI Pool / RoI Align ◄───┘
                                      │
                              Detection head
                              (class + box)
                                      │
                              Mask R-CNN only
                                      │
                                      ▼
                              Mask head (FCN)
```

---

## Key concepts

- **Region proposals** — candidate bounding boxes before classification  
- **Selective Search** — classical proposal algorithm (R-CNN, Fast R-CNN)  
- **RPN** — learned proposal network (Faster R-CNN, Mask R-CNN)  
- **RoI Pooling / RoI Align** — crop and resize features per proposal from a shared map  
- **Anchors** — reference boxes at multiple scales/ratios on the feature map  
- **NMS** — suppress duplicate detections  
- **Multi-task loss** — joint training of classification, regression, and (optionally) masks  
- **FPN** — multi-scale feature pyramid for objects of different sizes  
- **Instance segmentation** — per-object pixel masks (Mask R-CNN)  

## Output

- `Mask_RCNN.ipynb` saves `maskrcnn_pennfudan.pth` (model checkpoint)  
- Training plots and visualizations are produced inline in the notebooks  
- Generated checkpoints and Colab `data/` folders are excluded from git when stored under standard output paths  

## Further reading

- [R-CNN paper (2014)](https://arxiv.org/abs/1311.2524)  
- [Fast R-CNN paper (2015)](https://arxiv.org/abs/1504.08083)  
- [Faster R-CNN paper (2015)](https://arxiv.org/abs/1506.01497)  
- [Mask R-CNN paper (2017)](https://arxiv.org/abs/1703.06870)  
- [Torchvision detection models](https://pytorch.org/vision/stable/models.html#object-detection)  
