import time

from fastapi import FastAPI, Query

from recommender import load_model, recommend

app = FastAPI(title="Olist Recommender API")
model = load_model()  # โหลดครั้งเดียวตอนเปิดเซิร์ฟเวอร์


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model["model_name"],
        "trained_until": model["trained_until"],
    }


@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: str, k: int = Query(10, ge=1, le=50)):
    start = time.perf_counter()
    recs = recommend(model, customer_id, k)

    return {
        "customer_id": customer_id,
        "is_known_customer": customer_id in model["user_index"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "recommendations": recs,
    }