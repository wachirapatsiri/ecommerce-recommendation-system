# 🛒 E-Commerce Recommendation System

## 📌 Project Overview
An end-to-end item-based collaborative filtering recommender system built from scratch to analyze consumer purchasing behavior and improve product discovery. 

## 🎯 Business Problem
Customers often struggle to discover new products in large e-commerce catalogs. Relying purely on "Popularity" limits visibility for niche items. The goal of this project is to provide personalized recommendations that balance prediction accuracy with catalog diversity, driving cross-selling opportunities.

## 🛠️ Solution & Action (Tech Stack: Python, Pandas, Scikit-Learn)
* **Data Engineering:** Built a robust data pipeline to clean and merge complex relational datasets (Customers, Orders, Products, Reviews).
* **Modeling:** Developed an Item-Based Collaborative Filtering (Item-CF) algorithm trained on **112K** transaction records.
* **Evaluation:** Shifted away from standard accuracy metrics to use ranking metrics (HitRate@10) and Item Coverage to measure real-world recommendation value.

## 📈 Key Results
* Achieved a **HitRate@10 of 0.0238**, maintaining baseline accuracy while significantly outperforming the popularity model in diversity.
* Improved Item Coverage to **>1.5%**, ensuring a wider variety of products are surfaced to users rather than just top-sellers.

## Web : https://olist-recommender-wachirapat.streamlit.app/

## 💻 Live Demo & UI
<img width="1459" height="770" alt="image" src="https://github.com/user-attachments/assets/22b8c38f-69cb-4de9-813b-01fb565dc6bd" />
