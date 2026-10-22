# 🛒 E-Commerce Recommendation System (Olist Dataset)

## 📌 Project Overview
A machine learning-based recommendation engine developed using the Brazilian E-Commerce Public Dataset (Olist) to suggest relevant products to users based on historical purchasing behavior and market trends.

## 🎯 Business Problem
Providing generic product recommendations limits cross-selling opportunities and reduces user engagement. The challenge is to balance recommendations for "Cold Start" users (new customers) with highly personalized suggestions for returning customers to maximize the conversion rate.

## 🛠️ Solution & Action (Tech Stack: Python, Pandas, Matplotlib)
* **Data Engineering & Cleaning:** Cleansed and merged multiple datasets (orders, products, customers) handling missing values, temporal data, and ensuring referential integrity.
* **Exploratory Data Analysis (EDA):** Analyzed customer purchase breadth and top-selling categories to understand consumer economic behavior.
* **Recommendation Strategies:** Developed and compared multiple algorithms:
  1. All-time Popularity & Recent Popularity (for cold-start users).
  2. Collaborative Filtering (Item-CF) based on user interaction matrices.
  3. Hybrid Approach blending CF with category-specific trends.

## 📈 Key Results
* Successfully evaluated model performance using standard metrics like **HitRate@10** and **Item Coverage**.
* Demonstrated that a Hybrid approach effectively balances recommendation accuracy with product diversity, ensuring users discover both popular items and niche products.

## Web : https://olist-recommender-wachirapat.streamlit.app/
