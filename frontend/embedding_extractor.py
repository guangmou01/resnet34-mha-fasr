# PATH: frontend/embedding_extractor.py

import numpy as np
import onnxruntime as ort

from frontend.fbank import get_fbank

class EmbeddingExtractor:
    def __init__(self,
                 onnx_path: str,
                 device: str = "cpu",
                 embedding_type: str = "r-vector"):
        """
        :param onnx_path: Path to the ONNX model.
        :param device: "cpu" or "cuda"
        :param embedding_type: "r-vector" or "x-vector"
        """
        self.onnx_path = onnx_path
        self.device = device
        self.embedding_type = embedding_type

        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

    def extract(self,
                audio_path: str) -> np.ndarray:
        """
        :param audio_path: Path to the audio file.

        :return embedding: Extracted embedding, dtype float32.
        """
        fbank = get_fbank(audio_path)                         # [1, T, 64]
        fbank = fbank.numpy().astype(np.float32)

        if self.embedding_type == "x-vector":
            fbank = np.transpose(fbank, (0, 2, 1))      # [1, 64, T]

        embedding = self.session.run(
            None,
            {self.input_name: fbank}
        )[0][0]

        return embedding.astype(np.float32)