import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class NLUProblemParser:
    """
    Phase 1: Natural Language Understanding & Problem Parsing Core.
    Parses conversational raw queries into machine-actionable technical specs.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        confidence_threshold: float = 0.7,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold

        # Define the 5 explicit data science problem tracks
        self.task_categories = [
            "Classification",
            "Regression",
            "Clustering",
            "Time-Series",
            "Anomaly Detection",
        ]

        self.reference_embeddings = self._generate_reference_embeddings()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Helper to generate sentence-transformer embeddings."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Apply mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def _generate_reference_embeddings(self) -> dict:
        """Pre-computes anchor embeddings for the 5 target categories."""
        anchors = {
            "Classification": "predict categorical targets fraud churn binary label class",
            "Regression": "predict continuous numerical target values house price estimation",
            "Clustering": "unsupervised group customer segmentation data clustering patterns",
            "Time-Series": "timestamped sequences forecasting future trends stock window history",
            "Anomaly Detection": "outlier detection rare events isolation industrial sensor failure",
        }
        return {cat: self._get_embedding(desc) for cat, desc in anchors.items()}

    def classify_intent(self, query: str) -> dict:
        """Matches user queries to a core task track using cosine similarity mapping."""
        query_emb = self._get_embedding(query)
        scores = {}

        # Calculate cosine similarity across reference anchors
        for cat, ref_emb in self.reference_embeddings.items():
            dot_prod = np.dot(query_emb, ref_emb)
            norm_prod = np.linalg.norm(query_emb) * np.linalg.norm(ref_emb)
            scores[cat] = float(dot_prod / norm_prod)

        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]

        status = "SUCCESS" if confidence >= self.confidence_threshold else "CLARIFICATION_REQUIRED"

        return {
            "query": query,
            "predicted_task": best_match,
            "confidence_score": round(confidence, 3),
            "status": status,
            "message": (
                "Task successfully routed."
                if status == "SUCCESS"
                else "Confidence below 0.7. Requesting user clarification."
            ),
        }

    def extract_metadata(self, query: str) -> dict:
        """Extracts contextual tags, variables, and domain metrics from raw query."""
        query_lower = query.lower()
        extracted = {
            "target_keywords": [],
            "domain_context": "General Data Science",
            "detected_expertise": (
                "Expert"
                if "auc" in query_lower or "smote" in query_lower
                else "Beginner"
            ),
        }

        keywords = [
            "churn", "fraud", "predict", "forecast",
            "maintenance", "anomaly", "temperature",
        ]
        for kw in keywords:
            if kw in query_lower:
                extracted["target_keywords"].append(kw)

        if "churn" in query_lower or "customer" in query_lower:
            extracted["domain_context"] = "Telecom/Commercial"
        elif (
            "sensor" in query_lower
            or "motor" in query_lower
            or "maintenance" in query_lower
        ):
            extracted["domain_context"] = "Industrial Manufacturing / ABB Operations"

        return extracted
