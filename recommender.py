"""Core recommendation logic shared by the Streamlit app (app.py) and the FastAPI service (api.py).

The model file is plain data (sparse matrices, ids, a popularity ranking) produced by
notebooks/E_commerce_Recommendation_System.ipynb, so no notebook code is needed to load it.

Recommendation strategy
-----------------------
1. Item-based collaborative filtering: score every item by its cosine similarity to the items
   the customer already bought (items they already own are excluded).
2. Fill the remaining slots with recent popular items (last 90 days of training data).

Customers with no purchase history (the vast majority) therefore receive the popularity list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender_itemcf_v1.joblib"
MAX_K = 50


class Recommender:
    def __init__(self, artifacts: dict[str, Any]):
        self.art = artifacts
        self._users_with_cf: list[str] | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "Recommender":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ metadata
    def info(self) -> dict[str, Any]:
        a = self.art
        return {
            "model_name": a["model_name"],
            "trained_until": a["trained_until"],
            "recent_days": int(a["recent_days"]),
            "n_customers": len(a["user_index"]),
            "n_items": len(a["item_ids"]),
            "n_interactions": int(a["X"].nnz),
            "evaluation": a.get("evaluation", {}),
        }

    # ------------------------------------------------------------------ helpers
    def has_customer(self, customer_id: str) -> bool:
        return customer_id in self.art["user_index"]

    def has_item(self, product_id: str) -> bool:
        return product_id in self.art["item_index"]

    def category(self, product_id: str) -> str:
        return self.art["item_category"].get(product_id, "unknown")

    def purchase_history(self, customer_id: str) -> list[dict[str, str]]:
        u = self.art["user_index"].get(customer_id)
        if u is None:
            return []
        items = self.art["item_ids"][self.art["X"][u].indices]
        return [{"product_id": str(i), "category": self.category(i)} for i in items]

    def _seen(self, customer_id: str) -> set[str]:
        return {h["product_id"] for h in self.purchase_history(customer_id)}

    def _cf(self, customer_id: str, k: int) -> list[tuple[str, float]]:
        a = self.art
        u = a["user_index"].get(customer_id)
        if u is None:
            return []
        row = a["X"][u]
        scores = (row @ a["item_sim"]).toarray().ravel()
        scores[row.indices] = 0.0  # never recommend what the customer already bought
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        top = candidates[np.argsort(-scores[candidates], kind="mergesort")][:k]
        return [(str(a["item_ids"][i]), float(scores[i])) for i in top]

    # ------------------------------------------------------------------ public API
    def recommend(self, customer_id: str, k: int = 10) -> list[dict[str, Any]]:
        k = max(1, min(int(k), MAX_K))
        seen = self._seen(customer_id)
        recs: list[tuple[str, str, float | None]] = [
            (item, "similar_items", round(score, 4)) for item, score in self._cf(customer_id, k)
        ]
        chosen = {item for item, _, _ in recs}
        for item in self.art["pop_ranking"]:
            if len(recs) >= k:
                break
            if item not in seen and item not in chosen:
                recs.append((item, "popular_recent", None))
                chosen.add(item)
        return [
            {"rank": r, "product_id": item, "category": self.category(item),
             "source": source, "score": score}
            for r, (item, source, score) in enumerate(recs, start=1)
        ]

    def similar_items(self, product_id: str, k: int = 10, min_shared_buyers: int = 1) -> list[dict[str, Any]]:
        """Items frequently bought by the same customers as `product_id` (item-to-item)."""
        a = self.art
        i = a["item_index"].get(product_id)
        if i is None:
            return []
        k = max(1, min(int(k), MAX_K))
        row = a["item_sim"][i]
        if row.nnz == 0:
            return []
        X = a["X"].tocsc()
        buyers = X[:, i]
        out = []
        for j, sim in sorted(zip(row.indices, row.data), key=lambda t: -t[1]):
            shared = int(buyers.multiply(X[:, j]).nnz)
            if shared < min_shared_buyers:
                continue
            pid = str(a["item_ids"][j])
            out.append({"product_id": pid, "category": self.category(pid),
                        "similarity": round(float(sim), 4), "shared_buyers": shared})
            if len(out) == k:
                break
        return out

    # ------------------------------------------------------------------ demo helpers
    def example_customers(self, n: int = 20) -> list[str]:
        """Customers with purchase history for whom item-CF finds at least one similar item."""
        if self._users_with_cf is None:
            a = self.art
            multi = np.flatnonzero(np.diff(a["X"].indptr) >= 2)
            inv = {v: k for k, v in a["user_index"].items()}
            found = []
            for u in multi:
                cid = inv[int(u)]
                if self._cf(cid, 1):
                    found.append(cid)
                if len(found) >= 200:
                    break
            self._users_with_cf = found
        return self._users_with_cf[:n]

    def example_items(self, n: int = 20) -> list[str]:
        """Items that have at least one similar item."""
        a = self.art
        with_neighbors = np.flatnonzero(np.diff(a["item_sim"].indptr) > 0)
        return [str(a["item_ids"][i]) for i in with_neighbors[:n]]
