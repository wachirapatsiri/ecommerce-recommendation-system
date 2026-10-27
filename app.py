"""Streamlit demo for the Olist recommender.   Run:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from recommender import Recommender

st.set_page_config(page_title="Olist Recommender", page_icon="🛒", layout="wide")

# Popularity baseline numbers come from the notebook (Section 9 / 10); the Item-CF numbers are
# read from the model file itself.
BASELINE = {"warm_hit_rate": 0.0237, "rolling_hit_rate": 0.0225}

TEXT = {
    "en": {
        "title": "🛒 Olist Product Recommender",
        "subtitle": "Item-based collaborative filtering with a recent-popularity fallback, "
                    "trained on the Brazilian Olist e-commerce dataset (2016-2018).",
        "lang": "Language",
        "tab_rec": "Recommend for a customer",
        "tab_sim": "Similar products",
        "tab_eval": "Evaluation & limits",
        "mode": "Customer type",
        "mode_ex": "Existing customer (example)",
        "mode_id": "Enter a customer ID",
        "mode_new": "New customer (no history)",
        "pick": "Customer",
        "enter_id": "Customer ID",
        "n_rec": "Number of recommendations",
        "go": "Recommend",
        "history": "Purchase history",
        "recs": "Recommendations",
        "no_history": "No purchase history found, so the list is the recent-popularity ranking.",
        "unknown_id": "This ID is not in the model. Showing the popularity list instead.",
        "src_cf": "similar to past purchases",
        "src_pop": "popular recently",
        "pick_item": "Product",
        "enter_item": "Product ID",
        "n_sim": "Number of similar products",
        "find": "Find similar",
        "sim_note": "'Shared buyers' is how many customers bought both products. "
                    "Small numbers mean weak evidence.",
        "no_sim": "No similar products found for this item.",
        "unknown_item": "Unknown product ID.",
        "model_info": "Model",
        "customers": "Customers",
        "products": "Products",
        "interactions": "Interactions",
        "trained": "Data until",
        "eval_title": "Offline evaluation (temporal split, HitRate@10)",
        "eval_cols": ["Setting", "Warm users", "Item-CF + Popularity", "Popularity (recent 90d)"],
        "eval_main": "Single split (cutoff {cutoff})",
        "eval_roll": "3 rolling splits (pooled)",
        "eval_cov": "Item coverage (single split)",
        "eval_text": (
            "- Item-CF + Popularity **matches** the popularity baseline on HitRate; the difference "
            "is **not statistically significant** (bootstrap CIs overlap and the paired difference "
            "includes 0).\n"
            "- Its advantage is **diversity**: it recommends far more distinct products than "
            "plain popularity.\n"
            "- 98% of evaluated customers had no history (cold start), and 94% of buyers bought a "
            "single product, so collaborative filtering only applies to a small group.\n"
            "- Offline evaluation only: no A/B test, so no claim about sales impact."
        ),
        "data_note": "Data: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0). "
                     "For learning and portfolio use.",
    },
    "th": {
        "title": "🛒 ระบบแนะนำสินค้า Olist",
        "subtitle": "Item-based collaborative filtering เติมด้วยสินค้ายอดนิยมล่าสุด "
                    "ฝึกจากข้อมูล Olist (ปี 2016-2018)",
        "lang": "ภาษา",
        "tab_rec": "แนะนำสินค้าให้ลูกค้า",
        "tab_sim": "สินค้าที่คล้ายกัน",
        "tab_eval": "ผลประเมินและข้อจำกัด",
        "mode": "ประเภทลูกค้า",
        "mode_ex": "ลูกค้าเก่า (ตัวอย่าง)",
        "mode_id": "พิมพ์ Customer ID",
        "mode_new": "ลูกค้าใหม่ (ไม่มีประวัติ)",
        "pick": "ลูกค้า",
        "enter_id": "Customer ID",
        "n_rec": "จำนวนสินค้าที่แนะนำ",
        "go": "แนะนำสินค้า",
        "history": "ประวัติการซื้อ",
        "recs": "รายการแนะนำ",
        "no_history": "ไม่พบประวัติการซื้อ จึงแสดงสินค้ายอดนิยมล่าสุด",
        "unknown_id": "ไม่พบ ID นี้ในโมเดล แสดงสินค้ายอดนิยมแทน",
        "src_cf": "คล้ายสินค้าที่เคยซื้อ",
        "src_pop": "ยอดนิยมล่าสุด",
        "pick_item": "สินค้า",
        "enter_item": "Product ID",
        "n_sim": "จำนวนสินค้าที่คล้ายกัน",
        "find": "ค้นหา",
        "sim_note": "'Shared buyers' คือจำนวนลูกค้าที่ซื้อสินค้าทั้งสองชิ้น ตัวเลขน้อยแปลว่าหลักฐานอ่อน",
        "no_sim": "ไม่พบสินค้าที่คล้ายกับชิ้นนี้",
        "unknown_item": "ไม่พบ Product ID นี้",
        "model_info": "โมเดล",
        "customers": "ลูกค้า",
        "products": "สินค้า",
        "interactions": "Interactions",
        "trained": "ข้อมูลถึง",
        "eval_title": "ผลประเมินแบบ Offline (แบ่งตามเวลา, HitRate@10)",
        "eval_cols": ["การทดสอบ", "Warm users", "Item-CF + Popularity", "Popularity (90 วันล่าสุด)"],
        "eval_main": "แบ่งครั้งเดียว (cutoff {cutoff})",
        "eval_roll": "แบ่ง 3 ช่วงเวลา (รวมผล)",
        "eval_cov": "Item coverage (แบ่งครั้งเดียว)",
        "eval_text": (
            "- Item-CF + Popularity ได้ HitRate **เท่ากับ** Popularity และความต่าง**ไม่มีนัยสำคัญ"
            "ทางสถิติ** (CI ซ้อนทับกัน และช่วงความต่างคร่อม 0)\n"
            "- จุดเด่นคือ **ความหลากหลาย** แนะนำสินค้าต่างชนิดมากกว่า Popularity อย่างเดียวมาก\n"
            "- ผู้ถูกประเมิน 98% ไม่มีประวัติ (Cold Start) และผู้ซื้อ 94% ซื้อสินค้าเพียงชนิดเดียว "
            "CF จึงใช้ได้กับกลุ่มเล็กๆ เท่านั้น\n"
            "- เป็นการประเมินแบบ Offline ยังไม่มี A/B test จึงไม่สรุปผลต่อยอดขาย"
        ),
        "data_note": "ข้อมูล: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0) "
                     "ใช้เพื่อการเรียนรู้และพอร์ตโฟลิโอ",
    },
}


@st.cache_resource(show_spinner="Loading model...")
def get_model() -> Recommender:
    return Recommender.load()


def short(pid: str) -> str:
    return f"{pid[:8]}…"


rec = get_model()
info = rec.info()
ev = info["evaluation"]

lang = st.sidebar.radio("Language / ภาษา", ["en", "th"], format_func=lambda x: {"en": "English", "th": "ไทย"}[x])
t = TEXT[lang]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader(t["model_info"])
    st.caption(info["model_name"])
    st.metric(t["customers"], f"{info['n_customers']:,}")
    st.metric(t["products"], f"{info['n_items']:,}")
    st.metric(t["interactions"], f"{info['n_interactions']:,}")
    st.caption(f"{t['trained']}: {str(info['trained_until'])[:10]}")
    if ev:
        lo, hi = ev["warm_hit_rate_ci95"]
        st.caption(
            f"HitRate@10 (warm, n={ev['warm_users']}): {ev['warm_hit_rate@10']:.4f} "
            f"(95% CI {lo:.3f}-{hi:.3f})"
        )
    st.caption(t["data_note"])

st.title(t["title"])
st.caption(t["subtitle"])

tab_rec, tab_sim, tab_eval = st.tabs([t["tab_rec"], t["tab_sim"], t["tab_eval"]])


def render_recs(df: pd.DataFrame) -> None:
    df = df.copy()
    df["source"] = df["source"].map({"similar_items": t["src_cf"], "popular_recent": t["src_pop"]})
    df["score"] = df["score"].map(lambda s: "" if pd.isna(s) else f"{s:.3f}")
    st.dataframe(df, hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- tab 1
with tab_rec:
    mode = st.radio(t["mode"], ["ex", "id", "new"], horizontal=True,
                    format_func=lambda m: t[f"mode_{m}"])
    customer_id = ""
    if mode == "ex":
        examples = rec.example_customers(20)
        if examples:
            customer_id = st.selectbox(t["pick"], examples, format_func=short)
    elif mode == "id":
        customer_id = st.text_input(t["enter_id"]).strip()
    else:
        customer_id = "new_customer_demo"
    k = st.slider(t["n_rec"], 1, 20, 10, key="k_rec")

    if st.button(t["go"], type="primary", disabled=not customer_id):
        history = rec.purchase_history(customer_id)
        if mode == "id" and not history:
            st.warning(t["unknown_id"])
        elif not history:
            st.info(t["no_history"])
        col_h, col_r = st.columns([1, 2])
        with col_h:
            st.subheader(t["history"])
            if history:
                st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        with col_r:
            st.subheader(t["recs"])
            render_recs(pd.DataFrame(rec.recommend(customer_id, k)))

# ----------------------------------------------------------------------------- tab 2
with tab_sim:
    st.caption(t["sim_note"])
    SIM_MODES = {
        "ex": {"en": "Example product", "th": "สินค้าตัวอย่าง"},
        "id": {"en": "Enter a product ID", "th": "พิมพ์ Product ID"},
    }
    src = st.radio(t["pick_item"], ["ex", "id"], horizontal=True, key="sim_mode",
                   format_func=lambda m: SIM_MODES[m][lang])
    product_id = ""
    if src == "ex":
        product_id = st.selectbox(t["pick_item"], rec.example_items(20),
                                  format_func=lambda p: f"{short(p)} · {rec.category(p)}")
    else:
        product_id = st.text_input(t["enter_item"]).strip()
    n_sim = st.slider(t["n_sim"], 1, 20, 5, key="k_sim")
    if st.button(t["find"], disabled=not product_id):
        if not rec.has_item(product_id):
            st.warning(t["unknown_item"])
        else:
            sims = rec.similar_items(product_id, n_sim)
            if sims:
                st.dataframe(pd.DataFrame(sims), hide_index=True, width="stretch")
            else:
                st.info(t["no_sim"])

# ----------------------------------------------------------------------------- tab 3
with tab_eval:
    st.subheader(t["eval_title"])
    if ev:
        c = t["eval_cols"]
        table = pd.DataFrame([
            {c[0]: t["eval_main"].format(cutoff=ev["cutoff"]), c[1]: ev["warm_users"],
             c[2]: f"{ev['warm_hit_rate@10']:.4f}", c[3]: f"{BASELINE['warm_hit_rate']:.4f}"},
            {c[0]: t["eval_roll"], c[1]: ev["rolling_warm_cases"],
             c[2]: f"{ev['rolling_hit_rate@10']:.4f}", c[3]: f"{BASELINE['rolling_hit_rate']:.4f}"},
            {c[0]: t["eval_cov"], c[1]: ev["warm_users"],
             c[2]: f"{ev['warm_item_coverage'] * 100:.2f}%", c[3]: f"{ev['pop_item_coverage'] * 100:.2f}%"},
        ])
        st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(t["eval_text"])
