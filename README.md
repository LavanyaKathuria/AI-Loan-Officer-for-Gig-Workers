# 🤖 AI Loan Officer for Gig Workers

An AI-powered web application that evaluates loan applications for gig workers based on their earning patterns, work stability, platform tenure, and financial behavior — just like a real loan officer would, but faster, more consistent, and without bias.

Hugging Face Space: [https://huggingface.co/spaces/YourSpaceName](https://huggingface.co/spaces/Lavanyakathuria/AI-Loan-Officer-Gig-Workers)
---

## 📌 Table of Contents
- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Logic](#model-logic)
- [UI](#ui)


## 📖 About
Gig workers often face challenges when applying for loans due to irregular income and lack of formal credit history.  
Our **AI Loan Officer** uses a custom-trained machine learning model to **predict repayment probability**, make **loan approval/rejection decisions**, and even **recommend loan terms and interest rates** tailored to each applicant’s profile.

---

## ✨ Features
✅ **Loan Approval Prediction** – Approves or rejects based on repayment probability.  
✅ **Custom Loan Terms** – Suggests optimal loan term & interest rate based on applicant risk profile.  
✅ **Explainable AI** – Provides reasoning behind approvals/rejections using SHAP-based feature importance.  
✅ **Real-Time Processing** – Predictions within milliseconds for a smooth user experience.  
✅ **Dynamic UI** – Clean, responsive, and professional Gradio interface.

---

## 🛠 Tech Stack
**Frontend/UI:** Gradio (Python)  
**Backend/ML:** Scikit-learn, Pandas, NumPy  
**Visualization:** Matplotlib, SHAP  
**Language:** Python 3.10+  
**Environment:** Virtualenv / Conda  

---

## 📊 Dataset
A synthetic-yet-realistic dataset of **20,000+ gig workers** generated with a lot of research and carefully designed interdependencies to reflect real-world conditions.  
Top 15 most important features (in order of weight):

1. avg_monthly_earnings  
2. total_earnings  
3. earnings_trend_slope_6mo  
4. savings  
5. rating  
6. working_hours_per_day  
7. tenure_platform_1_months  
8. tenure_platform_2_months  
9. platform_primary  
10. sector  
11. loan_amount_requested  
12. location_stability  
13. recurring_expenses  
14. cash_inflow_to_outflow_ratio  
15. credit_history_length_months  

---

## 🧠 Model Logic
1. **Input Features** from applicant profile.  
2. **Preprocessing Pipeline** – Handles missing values, encoding, and scaling.  
3. **Random Forest Classifier** – Trained to predict repayment probability.  
4. **Decision Layer** –  
   - `repayment_probability ≥ threshold → APPROVE`  
   - Else → REJECT  
5. **Term & Interest Recommendation** – Adjusted based on predicted risk category.  
6. **Explainability Module** – Displays which factors influenced the decision.

---

## 🎨 UI
The Gradio interface includes:
- **Clean, soft-color theme** for professional look
- Input sliders/dropdowns for applicant details
- **Instant results** with approval/rejection message
- Loan term & interest rate recommendations
- Feature importance visualization

---


# Install dependencies
pip install -r requirements.txt
