# PATENT DIAGRAMS FOR LEGAL SUBMISSION

## Figure 1: Dual-Phase Biometric System Architecture
*Referenced in [0009] and [0017-0020]*

```mermaid
graph TB
    subgraph "100 - DUAL-PHASE PROCESSING PIPELINE"
        subgraph "102 - INPUT LAYER"
            101[Video Input<br/>Camera Source] --> 103[Face Detection<br/>Component]
            101 --> 104[Person Detection<br/>Component]
        end
        
        subgraph "110 - PHASE 1: PHYSIOLOGICAL SCANNER"
            103 --> 111[MediaPipe Face Mesh<br/>468 Landmarks]
            111 --> 112[Demographic Analysis<br/>Age/Gender/Ethnicity]
            111 --> 113[PPG Heart Rate<br/>Analysis]
            111 --> 114[Thermal Temperature<br/>Analysis]
            113 --> 115[Blood Pressure<br/>Pulse Morphology]
        end
        
        subgraph "120 - PHASE 2: BODY ESTIMATOR"
            104 --> 121[YOLOv8 Person<br/>Detection]
            121 --> 122[MediaPipe Pose<br/>33 Landmarks]
            121 --> 123[MiDaS Depth<br/>Estimation]
            121 --> 124[SAM Body<br/>Segmentation]
            122 --> 125[SMPL-X Model<br/>Fitting]
            123 --> 126[Height Estimation<br/>Multi-Method]
            125 --> 127[Weight Estimation<br/>Multi-Factor]
            126 --> 128[Result Validation<br/>Outlier Detection]
            127 --> 128
        end
        
        subgraph "140 - CONFIGURATION BRIDGE"
            112 --> 141[Biometric Profile<br/>Creation]
            113 --> 141
            114 --> 141
            115 --> 141
            141 --> 142[Algorithm<br/>Calibration]
            142 --> 126
            142 --> 127
        end
        
        subgraph "130 - RLHF SYSTEM"
            128 --> 131[Ground Truth<br/>Collection]
            131 --> 132[Data Quality<br/>Assessment]
            132 --> 133[Training Data<br/>Preparation]
            133 --> 134[Model Update<br/>Coordination]
            134 --> 142
        end
    end
    
    128 --> 150[Final Biometric<br/>Results Output]
```

## Figure 2: Enhanced Physiological Scanner Workflow
*Referenced in [0010] and [0021-0024]*

```mermaid
flowchart TD
    201[Facial Video Input] --> 202[MediaPipe Face Mesh<br/>468 Landmarks]
    
    202 --> 203[ROI Extraction<br/>Region Analysis]
    203 --> 204[Cheek Region<br/>Selection]
    203 --> 205[Forehead Region<br/>Selection]
    203 --> 206[Periorbital Region<br/>Selection]
    
    subgraph "210 - PPG SIGNAL PROCESSING"
        204 --> 211[Green Channel<br/>Isolation]
        211 --> 212[Butterworth Filter<br/>0.7-4.0 Hz]
        212 --> 213[FFT Analysis<br/>Frequency Domain]
        213 --> 214[Peak Detection<br/>Physiological Range]
        214 --> 215[Heart Rate<br/>BPM Calculation]
    end
    
    subgraph "220 - PULSE MORPHOLOGY"
        215 --> 221[Systolic Peak<br/>Detection]
        221 --> 222[Dicrotic Notch<br/>Analysis]
        222 --> 223[Blood Pressure<br/>Estimation]
    end
    
    subgraph "230 - THERMAL ANALYSIS"
        205 --> 231[Color Space<br/>Conversion]
        231 --> 232[Vascular Pattern<br/>Detection]
        232 --> 233[Thermal Signature<br/>Mapping]
        233 --> 234[Temperature<br/>Estimation]
    end
    
    subgraph "240 - DEMOGRAPHIC AI"
        206 --> 241[Facial Feature<br/>Extraction]
        241 --> 242[Age Estimation<br/>CNN Model]
        242 --> 243[Gender Classification<br/>Deep Learning]
        243 --> 244[Ethnicity Analysis<br/>Multi-Modal AI]
    end
    
    215 --> 250[Confidence Assessment<br/>& Quality Scoring]
    223 --> 250
    234 --> 250
    244 --> 250
    
    250 --> 260[Physiological Profile<br/>Generation]
```

## Figure 3: Advanced Body Estimator Pipeline
*Referenced in [0011] and [0025-0029]*

```mermaid
graph TD
    300[Person Detection Input] --> 301[Stage 1: YOLOv8<br/>Person Detection]
    
    301 --> 302[Stage 2: MediaPipe<br/>Pose Processing]
    301 --> 303[Stage 3: MiDaS<br/>Depth Estimation]
    301 --> 304[Stage 4: SAM<br/>Body Segmentation]
    
    302 --> 305[Stage 5: SMPL-X<br/>Model Fitting]
    303 --> 305
    304 --> 305
    
    subgraph "306 - HEIGHT ESTIMATION METHODS"
        302 --> 306a[Method 1:<br/>Landmark Distance<br/>30% Weight]
        303 --> 306b[Method 2:<br/>Depth Correction<br/>50% Weight]
        305 --> 306c[Method 3:<br/>SMPL-X Validation<br/>20% Weight]
        
        306a --> 306d[Ensemble<br/>Height Prediction]
        306b --> 306d
        306c --> 306d
    end
    
    subgraph "307 - WEIGHT ESTIMATION FACTORS"
        305 --> 307a[Factor 1:<br/>Volume-Based<br/>SMPL-X Calculation]
        302 --> 307b[Factor 2:<br/>Anthropometric<br/>Proportions]
        309[Demographic Data] --> 307c[Factor 3:<br/>Demographic<br/>Adjustments]
        
        307a --> 307d[Ensemble<br/>Weight Prediction]
        307b --> 307d
        307c --> 307d
    end
    
    306d --> 308[Stage 8: Result<br/>Validation]
    307d --> 308
    
    308 --> 310[Final Measurements<br/>Height + Weight + Confidence]
```

## Figure 4: PPG Signal Processing Algorithm
*Referenced in [0012] and [0024]*

```mermaid
flowchart TD
    401[Facial Video<br/>Sequence Input] --> 402[ROI Extraction<br/>Cheek Regions]
    402 --> 403[Adaptive Region<br/>Sizing]
    403 --> 404[Skin Tone<br/>Filtering]
    
    404 --> 405[Green Channel<br/>Isolation]
    405 --> 406[Butterworth Filter<br/>0.7-4.0 Hz Bandpass]
    406 --> 407[Baseline Trend<br/>Removal]
    407 --> 408[Kalman Filtering<br/>Noise Reduction]
    
    408 --> 409[FFT Analysis<br/>Frequency Domain]
    409 --> 410[Peak Detection<br/>Physiological Range<br/>42-240 BPM]
    410 --> 411[Heart Rate Calculation<br/>Peak Frequency × 60]
    
    subgraph "420 - QUALITY ASSESSMENT"
        411 --> 421[Signal-to-Noise<br/>Ratio Analysis]
        421 --> 422[Peak Prominence<br/>Assessment]
        422 --> 423[Consistency<br/>Evaluation]
        423 --> 424[Confidence Score<br/>Generation]
    end
    
    424 --> 430[Heart Rate Output<br/>BPM + Confidence]
```

## Figure 5: RLHF Continuous Learning System
*Referenced in [0013] and [0030-0034]*

```mermaid
graph TB
    subgraph "500 - USER INTERACTION LAYER"
        501[Biometric Scan<br/>Results] --> 502[User Feedback<br/>Interface]
        502 --> 503[Ground Truth<br/>Input Collection]
        503 --> 504[Measurement<br/>Validation]
    end
    
    subgraph "510 - DATA COLLECTION SYSTEM"
        504 --> 511[Session Management<br/>Unique Identifiers]
        511 --> 512[Data Quality<br/>Assessment]
        512 --> 513[Outlier Detection<br/>& Flagging]
        513 --> 514[Confidence Rating<br/>Analysis]
    end
    
    subgraph "520 - PRIVACY & SECURITY"
        514 --> 521[Data Anonymization<br/>PII Removal]
        521 --> 522[AES-256 Encryption<br/>Secure Storage]
        522 --> 523[Differential Privacy<br/>Noise Addition]
    end
    
    subgraph "530 - TRAINING DATA PIPELINE"
        523 --> 531[Feature Extraction<br/>From Scan Data]
        531 --> 532[Measurement<br/>Normalization]
        532 --> 533[Physiological Vector<br/>Creation]
        533 --> 534[Training Label<br/>Generation]
    end
    
    subgraph "540 - MODEL ADAPTATION ENGINE"
        534 --> 541[Algorithm Weight<br/>Updates]
        541 --> 542[Demographic<br/>Calibration]
        542 --> 543[Physiological<br/>Correlation Refinement]
        543 --> 544[Ensemble Weight<br/>Optimization]
    end
    
    544 --> 550[Updated Algorithm<br/>Deployment]
    550 --> 501
```

## Figure 6: System Architecture & Performance Stack
*Referenced in [0014] and [0038-0041]*

```mermaid
graph TD
    subgraph "600 - HARDWARE LAYER"
        601[CPU Requirements<br/>Intel i5+ / AMD Ryzen 5+<br/>Multi-core Processing]
        602[Memory Specifications<br/>4GB Minimum<br/>16GB+ Optimal]
        603[Storage Requirements<br/>3GB Free Space<br/>SSD Recommended]
        604[Camera Hardware<br/>720p Minimum<br/>1080p Optimal]
    end
    
    subgraph "610 - COMPUTER VISION STACK"
        611[OpenCV 4.8.1.78<br/>Image Processing Core]
        612[MediaPipe 0.10.7<br/>Landmark Detection]
        613[Ultralytics 8.0.0<br/>YOLO Object Detection]
        614[Segment Anything 1.0<br/>Advanced Segmentation]
    end
    
    subgraph "620 - DEEP LEARNING LAYER"
        621[PyTorch 2.1.1<br/>Neural Network Engine]
        622[TensorFlow 2.14.0<br/>Model Support Framework]
        623[SMPL-X 0.1.28<br/>3D Body Modeling]
    end
    
    subgraph "630 - PERFORMANCE METRICS"
        631[Face Detection<br/>~15ms per frame]
        632[Physiological Analysis<br/>~200ms per scan]
        633[Body Processing<br/>30-60 seconds total]
        634[Real-time Processing<br/>30+ FPS capability]
    end
    
    601 --> 611
    602 --> 621
    603 --> 623
    604 --> 612
    
    611 --> 631
    612 --> 631
    613 --> 633
    621 --> 632
    622 --> 632
    623 --> 633
```

## Figure 7: Processing Pipeline Performance Map
*Referenced in [0015]*

```mermaid
gantt
    title System Processing Timeline - Single Scan Cycle
    dateFormat X
    axisFormat %Lms
    
    section Face Analysis
    MediaPipe Detection     :done, face1, 0, 15
    Landmark Extraction     :done, face2, 15, 25
    PPG Signal Processing   :done, face3, 25, 225
    
    section Body Analysis  
    YOLO Person Detection   :done, body1, 0, 25
    Pose Landmark Extract   :done, body2, 25, 50
    Depth Map Generation    :done, body3, 50, 200
    SAM Segmentation       :done, body4, 200, 350
    SMPL-X Model Fitting   :done, body5, 350, 500
    
    section Measurements
    Height Calculation     :done, meas1, 500, 520
    Weight Estimation      :done, meas2, 520, 540
    Result Validation      :done, meas3, 540, 560
    
    section Output
    Report Generation      :done, out1, 560, 580
    Confidence Scoring     :done, out2, 580, 600
```

## Figure 8: Privacy-Preserving Data Protection Architecture
*Referenced in [0016] and [0035-0037]*

```mermaid
graph TD
    subgraph "800 - LOCAL PROCESSING LAYER"
        801[Raw Biometric Data<br/>Video/Images] --> 802[Local AI Inference<br/>No Cloud Dependency]
        802 --> 803[On-Device Processing<br/>Complete User Control]
    end
    
    subgraph "810 - ENCRYPTION & SECURITY"
        803 --> 811[AES-256 Encryption<br/>Biometric Data]
        811 --> 812[User-Specific<br/>Encryption Keys]
        812 --> 813[Secure Storage<br/>Local Database]
    end
    
    subgraph "820 - PRIVACY PROTECTION"
        813 --> 821[Data Anonymization<br/>PII Removal]
        821 --> 822[Differential Privacy<br/>Noise Addition]
        822 --> 823[Personal Identifier<br/>Scrubbing]
    end
    
    subgraph "830 - DATA MANAGEMENT"
        823 --> 831[Retention Policy<br/>Automatic Cleanup]
        831 --> 832[Secure Deletion<br/>Cryptographic Erasure]
        832 --> 833[Audit Logging<br/>Access Tracking]
    end
    
    subgraph "840 - COMPLIANCE FRAMEWORK"
        833 --> 841[GDPR Compliance<br/>Right to Erasure]
        841 --> 842[Consent Management<br/>Granular Controls]
        842 --> 843[Data Portability<br/>Export Options]
    end
    
    subgraph "850 - OPTIONAL SHARING"
        843 --> 851[User Consent<br/>Explicit Permission]
        851 --> 852[Federated Learning<br/>Anonymous Participation]
        852 --> 853[Research Contribution<br/>De-identified Data]
    end
```

---

