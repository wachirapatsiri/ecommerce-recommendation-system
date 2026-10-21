import time

import numpy as np
import pandas as pd
import streamlit as st

from recommender import get_history, load_model, recommend

st.set_page_config(page_title="Olist Recommender Demo", layout="wide")


@st.cache_resource  # โหลดโมเดลครั้งเดียว ไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม
def load_everything():
    model = load_model()
    user_ids = list(model["user_index"])

    # หาลูกค้าตัวอย่างที่ซื้อ ≥ 2 ชิ้น และระบบหาสินค้าคล้ายได้
    multi = np.flatnonzero(model["X"].getnnz(axis=1) >= 2)[:300]
    demo = [
        user_ids[i] for i in multi
        if recommend(model, user_ids[i], 1)[0]["source"] == "similar_items"
    ][:20]
    return model, demo


model, demo_customers = load_everything()

# ---------- แถบด้านข้าง: ข้อมูลโมเดล ----------
with st.sidebar:
    st.header("ข้อมูลโมเดล")
    st.write(f"**โมเดล:** {model['model_name']}")
    st.write(f"**ข้อมูลถึง:** {model['trained_until'][:10]}")
    st.write(f"**ลูกค้า:** {len(model['user_index']):,} คน")
    st.write(f"**สินค้า:** {len(model['item_ids']):,} ชนิด")
    ev = model["evaluation"]
    st.write(f"**HitRate@10 (ลูกค้าเก่า):** {ev['warm_hit_rate@10']:.4f}")

# ---------- หน้าหลัก ----------
st.title("ระบบแนะนำสินค้า Olist")
st.caption("ลูกค้าเก่า: แนะนำสินค้าคล้ายของที่เคยซื้อ | ลูกค้าใหม่: แนะนำสินค้าขายดีล่าสุด")

mode = st.radio(
    "เลือกประเภทลูกค้า",
    ["ลูกค้าเก่า (ตัวอย่าง)", "ลูกค้าใหม่", "พิมพ์ Customer ID เอง"],
    horizontal=True,
)

if mode == "ลูกค้าเก่า (ตัวอย่าง)":
    customer_id = st.selectbox("เลือกลูกค้า", demo_customers)
elif mode == "ลูกค้าใหม่":
    customer_id = "new_customer_demo"
    st.info("ลูกค้าคนนี้ไม่มีประวัติการซื้อในระบบ")
else:
    customer_id = st.text_input("customer_unique_id").strip()

k = st.slider("จำนวนสินค้าที่แนะนำ", 5, 20, 10)

if st.button("แนะนำสินค้า", type="primary") and customer_id:
    start = time.perf_counter()
    recs = recommend(model, customer_id, k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    left, right = st.columns([1, 2])

    with left:
        st.subheader("ประวัติการซื้อ")
        history = get_history(model, customer_id)
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True)
        else:
            st.write("ไม่มีประวัติ (ลูกค้าใหม่)")

    with right:
        st.subheader("รายการแนะนำ")
        df = pd.DataFrame(recs)
        df["source"] = df["source"].map({
            "similar_items": "คล้ายของที่เคยซื้อ",
            "popular_recent": "ขายดีล่าสุด",
        })
        st.dataframe(df, hide_index=True)

    st.success(f"ใช้เวลา {elapsed_ms:.1f} ms")