"""
Hugging Face inference helper for driver-state classification.

Supports local Transformers zero-shot image classification and
hosted Hugging Face Inference API as a fallback backend.
"""

from __future__ import annotations

import io
import importlib
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


class HuggingFaceDrowsinessClassifier:
    """Classify Alert, Drowsy, and Yawning states using Hugging Face models."""

    STATE_ORDER: List[str] = ["Alert", "Drowsy", "Yawning"]

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        class_labels: Dict[str, str] | None = None,
        confidence_threshold: float = 0.35,
        backend: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.class_labels = class_labels or {
            "Alert": "an alert driver with eyes open",
            "Drowsy": "a drowsy sleepy driver with eyes closed",
            "Yawning": "a yawning driver with mouth open",
        }
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

        self.backend = ""
        self._local_pipeline = None
        self._client = None

        self._candidate_labels = [self.class_labels[state] for state in self.STATE_ORDER]
        self._label_to_state = {
            self.class_labels[state].strip().lower(): state for state in self.STATE_ORDER
        }

        self._initialize_backend(backend)

    def _initialize_backend(self, backend: str) -> None:
        mode = (backend or "auto").strip().lower()
        if mode not in {"auto", "local", "hosted"}:
            raise ValueError(f"Unsupported Hugging Face backend mode: {backend}")

        errors: List[str] = []

        if mode in {"auto", "local"}:
            try:
                pipeline = getattr(importlib.import_module("transformers"), "pipeline")

                self._local_pipeline = pipeline(
                    task="zero-shot-image-classification",
                    model=self.model_id,
                )
                self.backend = "local"
                return
            except Exception as exc:  # pragma: no cover - backend dependent
                errors.append(f"local backend error: {exc}")

        if mode in {"auto", "hosted"}:
            try:
                InferenceClient = getattr(importlib.import_module("huggingface_hub"), "InferenceClient")

                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                self._client = InferenceClient(provider="hf-inference", token=token)
                self.backend = "hosted"
                return
            except Exception as exc:  # pragma: no cover - backend dependent
                errors.append(f"hosted backend error: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "No Hugging Face backend available")

    @staticmethod
    def _to_pil_image(frame: np.ndarray) -> Image.Image:
        arr = np.asarray(frame)

        if arr.ndim == 3 and arr.shape[2] == 3:
            # Face crops in this project come from OpenCV in BGR order.
            arr = arr[:, :, ::-1]

        if arr.dtype != np.uint8:
            scale = 255.0 if np.max(arr) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    @staticmethod
    def _extract_label_score(item) -> Tuple[str, float]:
        if isinstance(item, dict):
            return str(item.get("label", "")), float(item.get("score", 0.0))

        label = getattr(item, "label", "")
        score = getattr(item, "score", 0.0)
        return str(label), float(score)

    def _infer_local(self, image: Image.Image):
        if self._local_pipeline is None:
            raise RuntimeError("Local Hugging Face backend is not initialized")

        try:
            return self._local_pipeline(
                image,
                candidate_labels=self._candidate_labels,
                hypothesis_template="This image shows {}.",
            )
        except TypeError:
            return self._local_pipeline(image, candidate_labels=self._candidate_labels)

    def _infer_hosted(self, image: Image.Image):
        if self._client is None:
            raise RuntimeError("Hosted Hugging Face backend is not initialized")

        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=92)
        image_bytes = payload.getvalue()

        call_variants = [
            {
                "image": image_bytes,
                "candidate_labels": self._candidate_labels,
                "model": self.model_id,
                "hypothesis_template": "This image shows {}.",
            },
            {
                "image": image_bytes,
                "labels": self._candidate_labels,
                "model": self.model_id,
                "hypothesis_template": "This image shows {}.",
            },
            {
                "image": image_bytes,
                "labels": self._candidate_labels,
                "model": self.model_id,
            },
        ]

        last_error = None
        for kwargs in call_variants:
            try:
                return self._client.zero_shot_image_classification(**kwargs)
            except TypeError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        raise RuntimeError("Hosted inference call failed")

    def _outputs_to_probs(self, outputs) -> np.ndarray:
        state_scores = {state: 0.0 for state in self.STATE_ORDER}

        for item in outputs:
            label, score = self._extract_label_score(item)
            normalized_label = label.strip().lower()
            state = self._label_to_state.get(normalized_label)

            if state is None:
                if "yawn" in normalized_label or "mouth open" in normalized_label:
                    state = "Yawning"
                elif "drows" in normalized_label or "sleep" in normalized_label or "eyes closed" in normalized_label:
                    state = "Drowsy"
                elif "alert" in normalized_label or "awake" in normalized_label or "eyes open" in normalized_label:
                    state = "Alert"

            if state is not None:
                state_scores[state] = max(state_scores[state], float(score))

        probs = np.array([state_scores[state] for state in self.STATE_ORDER], dtype=np.float32)
        total = float(np.sum(probs))
        if total <= 1e-6:
            return np.array([0.8, 0.1, 0.1], dtype=np.float32)

        probs /= total
        return probs

    def predict(self, frame: np.ndarray) -> Tuple[str, np.ndarray, float]:
        image = self._to_pil_image(frame)

        try:
            if self.backend == "local":
                outputs = self._infer_local(image)
            else:
                outputs = self._infer_hosted(image)
        except Exception:
            # Fail safe during transient backend/API errors.
            return "Alert", np.array([0.8, 0.1, 0.1], dtype=np.float32), 0.8

        probs = self._outputs_to_probs(outputs)
        pred_idx = int(np.argmax(probs))
        state = self.STATE_ORDER[pred_idx]
        confidence = float(probs[pred_idx])

        if state != "Alert" and confidence < self.confidence_threshold:
            # Conservative fallback to avoid noisy non-alert predictions.
            probs = np.array([0.75, 0.15, 0.10], dtype=np.float32)
            state = "Alert"
            confidence = float(probs[0])

        return state, probs, confidence
