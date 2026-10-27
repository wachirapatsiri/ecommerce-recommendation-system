"""FastAPI service for the Olist recommender.

Run locally:   uvicorn api:app --reload
Docs (Swagger): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recommender import MAX_K, Recommender

state: dict[str, Recommender] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["rec"] = Recommender.load()
    yield
    state.clear()


app = FastAPI(
    title="Olist Recommender API",
    description="Item-based collaborative filtering with a recent-popularity fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


class Recommendation(BaseModel):
    rank: int
    product_id: str
    category: str
    source: str  # "similar_items" (collaborative filtering) or "popular_recent" (fallback)
    score: float | None = None


class RecommendationResponse(BaseModel):
    customer_id: str
    known_customer: bool  # False -> no purchase history, popularity list returned
    recommendations: list[Recommendation]


class SimilarItem(BaseModel):
    product_id: str
    category: str
    similarity: float
    shared_buyers: int  # how many customers bought both items (small numbers = weak evidence)


class SimilarItemsResponse(BaseModel):
    product_id: str
    category: str
    similar_items: list[SimilarItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return state["rec"].info()


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: str, k: int = Query(10, ge=1, le=MAX_K)):
    rec = state["rec"]
    return RecommendationResponse(
        customer_id=customer_id,
        known_customer=rec.has_customer(customer_id),
        recommendations=rec.recommend(customer_id, k),
    )


@app.get("/similar/{product_id}", response_model=SimilarItemsResponse)
def similar(product_id: str, k: int = Query(10, ge=1, le=MAX_K)):
    rec = state["rec"]
    if not rec.has_item(product_id):
        raise HTTPException(status_code=404, detail="Unknown product_id")
    return SimilarItemsResponse(
        product_id=product_id,
        category=rec.category(product_id),
        similar_items=rec.similar_items(product_id, k),
    )
