### FASR based on a ResNet34-MHA speaker embedding model

This repository stores and continuously updates software tools for likelihood-ratio-based forensic voice comparison / forensic automatic speaker recognition (FVC/FASR) using deep speaker representations (embeddings).

The toolkit currently includes modules for:

- frontend acoustic feature extraction; 
- DNN speaker embedding extraction; 
- backend likelihood-ratio scoring; 
- score calibration; 
- system validation.

We trained a speaker embedding extractor on VoxCeleb1+2 (7,205 spks). The model uses a ResNet34 backbone enhanced with multi-head attentive statistics pooling (MHA), referred to here as ResNet34-MHA. It achieved good performance in the ‘forensic_eval_01’ benchmark test (see `forensic_eval_01_demo.ipynb`).

### News & Development Plan

```text
# 2026.06.23    Initial repository setup.
```

We plan to continuously improve the functionality of our software toolkit. A technical description document will also be prepared to help practitioners and researchers from FVC community understand the system architecture, implementation details, and recommended usage.

### Usage

The architecture and trained parameters of the ResNet34-MHA were exported to ONNX format. In our demo, the expected model file is named `resnet34_mha.onnx`.

Because the .onnx file is relatively large, it is not distributed directly through this repository. We plan to share the model via Google Drive with interested practitioners and researchers.

If you would like to obtain access to the model, please contact the repository maintainer and briefly provide your name, affiliation, and intended use of the model:

```text
Deng, Guangmou
guangmou01@outlook.com
```
The recording files used in the experiments are also not distributed through this repository. Please obtain any required audio data through appropriate and authorised channels.

An expected project root should be:

```text
resnet34-mha-fasr/
├── frontend/
├── plda/
├── metric/
├── calibration/
├── forensic_eval_01/
├── resnet34_mha.onnx
├── forensic_eval_01_demo.ipynb
├── .gitignore
└── README.md
```

Install the required packages:

```bash
pip install -r requirements.txt
```



