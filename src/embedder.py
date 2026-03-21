"""Face embedding extraction using InsightFace/ArcFace models.

This module keeps embedding generation practical and integration-friendly:
- initialize model once
- optional lightweight preprocessing via FaceAligner
- simple API: embed(face_crop) -> embedding vector or None
- graceful handling for invalid/low-quality crops
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.face_aligner import FaceAligner

_INSIGHTFACE_MISSING_MESSAGE = (
    "InsightFace is not installed. Install insightface and onnxruntime in a supported "
    "Python environment, or add a fallback embedder."
)


class FaceEmbedderError(RuntimeError):
    """Raised when embedder initialization fails."""


class FaceEmbedder:
    """InsightFace/ArcFace embedding extractor.

    Expected config shape (subset):
        {
          "embedder_model": {
            "provider": "insightface" | "arcface",
            "model_name": "buffalo_l" | "path/to/model.onnx",
            "execution_provider": "CPUExecutionProvider" | "CUDAExecutionProvider"
          }
        }
    """

    def __init__(
        self,
        provider: str = "insightface",
        model_name: str = "buffalo_l",
        execution_provider: str = "CPUExecutionProvider",
        use_aligner: bool = True,
        min_face_size: int = 32,
        blur_threshold: float = 15.0,
    ) -> None:
        self.provider = provider.strip().lower()
        self.model_name = model_name.strip()
        self.execution_provider = execution_provider.strip()
        self.use_aligner = bool(use_aligner)
        self.min_face_size = int(min_face_size)
        self.blur_threshold = float(blur_threshold)

        if self.provider not in {"insightface", "arcface"}:
            raise FaceEmbedderError("provider must be 'insightface' or 'arcface'.")
        if not self.model_name:
            raise FaceEmbedderError("model_name cannot be empty.")
        if self.min_face_size < 16:
            raise FaceEmbedderError("min_face_size must be >= 16.")

        self._ensure_insightface_available()
        self._aligner = FaceAligner() if self.use_aligner else None
        self._mode: str
        self._model: Any
        self.last_debug_info: dict[str, Any] = {}
        self._initialize_model_once()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FaceEmbedder":
        """Build embedder from app config."""
        embed_cfg = config.get("embedder_model", {}) or {}
        return cls(
            provider=str(embed_cfg.get("provider", "insightface")),
            model_name=str(embed_cfg.get("model_name", "buffalo_l")),
            execution_provider=str(embed_cfg.get("execution_provider", "CPUExecutionProvider")),
            use_aligner=bool(config.get("use_face_aligner", True)),
            min_face_size=int(config.get("embedder_min_face_size", 32)),
            blur_threshold=float(config.get("embedder_blur_threshold", 15.0)),
        )

    def embed(self, face_crop: np.ndarray) -> np.ndarray | None:
        """Return embedding vector for one face crop.

        Args:
            face_crop: BGR face image crop as numpy array.

        Returns:
            1D float32 embedding vector, or None if crop is invalid/unusable.
        """
        self.last_debug_info = {}
        if not self._is_valid_crop(face_crop):
            self.last_debug_info = {
                "reason": "invalid_crop",
                "mode": getattr(self, "_mode", "unknown"),
            }
            return None
        if self._is_low_quality(face_crop):
            self.last_debug_info = {
                "reason": "low_quality_crop",
                "mode": getattr(self, "_mode", "unknown"),
                "crop_shape": list(face_crop.shape),
            }
            return None

        try:
            processed = self._preprocess(face_crop)
            if self._mode == "arcface_onnx":
                # ArcFaceONNX-like model expects RGB uint8 images.
                rgb = self._to_rgb_uint8(processed)
                feat = self._model.get_feat(rgb)
                emb = self._normalize_embedding(feat)
                if emb is None:
                    self.last_debug_info = {
                        "reason": "invalid_arcface_embedding",
                        "mode": self._mode,
                        "crop_shape": list(face_crop.shape),
                    }
                    return None
                self.last_debug_info = {
                    "reason": "ok",
                    "mode": self._mode,
                    "crop_shape": list(face_crop.shape),
                }
                return emb

            # FaceAnalysis fallback mode:
            # Prefer direct recognition model on the already cropped/aligned face.
            # This avoids re-detecting face inside a tight crop, which can fail frequently.
            rgb = self._to_rgb_uint8(processed)
            recognition_model = getattr(self._model, "models", {}).get("recognition")
            if recognition_model is not None and hasattr(recognition_model, "get_feat"):
                feat = recognition_model.get_feat(rgb)
                emb = self._normalize_embedding(feat)
                if emb is None:
                    self.last_debug_info = {
                        "reason": "invalid_face_analysis_feat",
                        "mode": self._mode,
                        "crop_shape": list(face_crop.shape),
                    }
                    return None
                self.last_debug_info = {
                    "reason": "ok",
                    "mode": self._mode,
                    "path": "face_analysis_recognition_model",
                    "crop_shape": list(face_crop.shape),
                }
                return emb

            faces = self._model.get(rgb)
            if not faces:
                self.last_debug_info = {
                    "reason": "face_analysis_no_faces_in_crop",
                    "mode": self._mode,
                    "crop_shape": list(face_crop.shape),
                }
                return None
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = getattr(best, "normed_embedding", None) or getattr(best, "embedding", None)
            if embedding is None:
                self.last_debug_info = {
                    "reason": "face_analysis_missing_embedding",
                    "mode": self._mode,
                    "crop_shape": list(face_crop.shape),
                }
                return None
            emb = self._normalize_embedding(embedding)
            if emb is None:
                self.last_debug_info = {
                    "reason": "face_analysis_invalid_embedding",
                    "mode": self._mode,
                    "crop_shape": list(face_crop.shape),
                }
                return None
            self.last_debug_info = {
                "reason": "ok",
                "mode": self._mode,
                "path": "face_analysis_get_faces",
                "crop_shape": list(face_crop.shape),
            }
            return emb
        except Exception as exc:
            # Keep pipeline robust; caller can skip this sample and continue.
            self.last_debug_info = {
                "reason": "embed_exception",
                "mode": getattr(self, "_mode", "unknown"),
                "error": str(exc),
                "crop_shape": list(face_crop.shape) if isinstance(face_crop, np.ndarray) else None,
            }
            return None

    def _initialize_model_once(self) -> None:
        """Initialize one model instance for the process lifetime."""
        get_model, FaceAnalysis = self._import_insightface()

        # Try direct ArcFace ONNX model first when model_name looks like a file path.
        model_path = Path(self.model_name)
        if model_path.exists() and model_path.is_file():
            try:
                self._model = get_model(str(model_path), providers=[self.execution_provider])
                # ctx_id: -1 means CPU. CUDA provider can still run if configured.
                ctx_id = 0 if "cuda" in self.execution_provider.lower() else -1
                self._model.prepare(ctx_id=ctx_id)
                self._mode = "arcface_onnx"
                return
            except Exception as exc:
                raise FaceEmbedderError(f"Failed to initialize ONNX embedder model: {exc}") from exc

        # Otherwise use FaceAnalysis model pack name (e.g., buffalo_l).
        try:
            app = FaceAnalysis(
                name=self.model_name,
                providers=[self.execution_provider],
                allowed_modules=["detection", "recognition"],
            )
            ctx_id = 0 if "cuda" in self.execution_provider.lower() else -1
            app.prepare(ctx_id=ctx_id)
            self._model = app
            self._mode = "face_analysis"
        except Exception as exc:
            raise FaceEmbedderError(
                f"Failed to initialize FaceAnalysis model '{self.model_name}': {exc}"
            ) from exc

    @staticmethod
    def _import_insightface() -> tuple[Any, Any]:
        """Import InsightFace lazily to avoid module import-time failures."""
        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise FaceEmbedderError(_INSIGHTFACE_MISSING_MESSAGE) from exc
        return get_model, FaceAnalysis

    @classmethod
    def _ensure_insightface_available(cls) -> None:
        """Validate InsightFace availability before creating embedder instance."""
        cls._import_insightface()

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Apply optional lightweight preprocessing before embedding."""
        if not self._aligner:
            return face_crop

        h, w = face_crop.shape[:2]
        bbox = [0, 0, w, h]
        return self._aligner.preprocess(frame=face_crop, bbox=bbox)

    @staticmethod
    def _is_valid_crop(face_crop: np.ndarray) -> bool:
        if face_crop is None or not isinstance(face_crop, np.ndarray):
            return False
        if face_crop.ndim != 3 or face_crop.shape[2] != 3:
            return False
        h, w = face_crop.shape[:2]
        return h > 0 and w > 0

    def _is_low_quality(self, face_crop: np.ndarray) -> bool:
        """Basic quality gate to avoid noisy registrations/recognition."""
        h, w = face_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return True

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score < self.blur_threshold

    @staticmethod
    def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """Convert preprocessed image to RGB uint8 safely."""
        out = image
        if out.dtype != np.uint8:
            # Supports [0,1] or [-1,1] ranges from aligner normalization.
            min_v = float(np.min(out))
            max_v = float(np.max(out))
            if min_v >= 0.0 and max_v <= 1.0:
                out = (out * 255.0).clip(0, 255).astype(np.uint8)
            else:
                out = (((out + 1.0) * 127.5).clip(0, 255)).astype(np.uint8)

        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _normalize_embedding(embedding: Any) -> np.ndarray | None:
        """Convert embedding output to normalized 1D float32 vector."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return None
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            return None
        return vec / norm
