### FASR based on a ResNet34-MHA speaker embedding model

This repository stores and continuously updates software tools for likelihood-ratio-based forensic voice comparison / forensic automatic speaker recognition (FVC/FASR) using deep speaker representations (embeddings).

The toolkit currently includes modules for:

- frontend acoustic feature extraction (`frontend`);
- DNN speaker embedding extraction (`frontend`); 
- backend likelihood-ratio scoring (`plda`); 
- score calibration (`calibration`); 
- system validation (`metric`).

We trained a speaker embedding extractor on VoxCeleb1+2 (7205 speakers). The model uses a ResNet34 backbone (He et al., 2016) enhanced with multi-head attentive statistics pooling (MHA, see India et al., 2019), referred to here as ResNet34-MHA. It achieved good performance in the ‘forensic_eval_01’ benchmark test (see `forensic_eval_01_demo.ipynb`).

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
├── calibration/
├── metric/
├── forensic_eval_01/
├── resnet34_mha.onnx
├── forensic_eval_01_demo.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90
- India, M., Safari, P., & Hernando, J. (2019). Self multi-Head attention for speaker recognition. arXiv. https://doi.org/10.48550/arXiv.1906.09890
- Morrison, G. S., Enzinger, E., Daniel, R., Joaquín, G.-R., & Alicia, L.-D. (2020). Statistical models in forensic voice comparison. In D. L. Banks, K. Kafadar, D. H. Kaye, & M. Tackett, Handbook of Forensic Statistics (pp. 451–497). CRC Press.
- Morrison, G. S., & Enzinger, E. (2016). Multi-laboratory evaluation of forensic voice comparison systems under conditions reflecting those of a real forensic case (forensic_eval_01) — Introduction. Speech Communication, 85, 119–126. https://doi.org/10.1016/j.specom.2016.07.006
- Morrison, G. S., & Enzinger, E. (2019). Multi-laboratory evaluation of forensic voice comparison systems under conditions reflecting those of a real forensic case (forensic_eval_01) — Conclusion. Speech Communication, 112, 37–39. https://doi.org/10.1016/j.specom.2019.06.007






