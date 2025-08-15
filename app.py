import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from gradio.themes.utils import fonts

# --- 1. LOAD ARTIFACTS ---
CLASSIFIER_PATH = "best_loan_approval_pipeline.joblib"
INTEREST_MODEL_PATH = "interest_rate_model.joblib"
pipeline = None
interest_pipeline = None

try:
    pipeline = joblib.load(CLASSIFIER_PATH)
    interest_pipeline = joblib.load(INTEREST_MODEL_PATH)
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    explainer = shap.TreeExplainer(classifier)
    num_features = pipeline.named_steps['preprocessor'].transformers_[0][2]
    cat_features = pipeline.named_steps['preprocessor'].transformers_[1][2]
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
    TRANSFORMED_FEATURE_NAMES = num_features + list(ohe_feature_names)
    print("✅ Classifier and Interest Rate models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")

# --- 2. HELPER & DECISION FUNCTIONS ---

def engineer_features(df):
    """
    Creates powerful, context-rich features.
    Total Tenure is now the SUM and savings_log_capped is re-added.
    """
    df_eng = df.copy()
    
    for col in ['avg_monthly_earnings', 'loan_amount_requested', 'savings', 'recurring_expenses', 
                'tenure_platform_1_months', 'tenure_platform_2_months', 'rating', 'working_hours_per_day']:
        if col in df_eng.columns:
            df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)

    # --- RE-ADDED: Cap and Log Transform Savings ---
    capped_savings = df_eng['savings'].clip(upper=200000)
    df_eng['savings_log_capped'] = np.log1p(capped_savings)
    
    df_eng['avg_monthly_earnings'] = df_eng['avg_monthly_earnings'].clip(lower=1)
    df_eng['loan_to_income_ratio'] = df_eng['loan_amount_requested'] / df_eng['avg_monthly_earnings']
    
    # --- REVERTED: Total tenure is now the SUM of the two platforms ---
    df_eng['total_tenure_months'] = df_eng['tenure_platform_1_months'] + df_eng['tenure_platform_2_months']
    
    df_eng['disposable_income'] = df_eng['avg_monthly_earnings'] - df_eng['recurring_expenses']
    
    log_tenure = np.log1p(df_eng['total_tenure_months'])
    log_avg_earnings = np.log1p(df_eng['avg_monthly_earnings'])
    df_eng['gig_performance_score'] = (
        df_eng['rating'] * log_tenure * log_avg_earnings * df_eng['working_hours_per_day']
    ).fillna(0)
    
    return df_eng

# ... (The rest of the app.py script remains the same as the last version)
# calculate_emi, get_narrative_reasoning, apply_business_rules, make_loan_decision, and the Gradio UI
# do not need to change, as they adapt to the features provided by engineer_features.

def calculate_emi(principal, annual_rate, term_months):
    if principal <= 0 or annual_rate <= 0 or term_months <= 0: return 0
    monthly_rate = (annual_rate / 100) / 12
    emi = principal * monthly_rate * (1 + monthly_rate)**term_months / ((1 + monthly_rate)**term_months - 1)
    return emi

def get_narrative_reasoning(shap_values, feature_names, decision, top_n=3):
    sv = pd.Series(shap_values, index=feature_names)
    friendly_names = {name: name.replace('_', ' ').replace('num__', '').replace('cat__', '') for name in feature_names}
    sv.index = sv.index.map(friendly_names)
    pos_contributors = sv[sv > 0].nlargest(top_n)
    neg_contributors = sv[sv < 0].nsmallest(top_n)
    strengths = ", ".join(pos_contributors.index) if not pos_contributors.empty else "None identified"
    concerns = ", ".join(neg_contributors.index) if not neg_contributors.empty else "None identified"
    summary = "This was a borderline case with competing positive and negative factors."
    if sv.sum() > 0.5: summary = "Overall, the applicant's positive factors significantly outweigh the risks."
    elif sv.sum() < -0.5: summary = "Overall, the model identified several risk factors that led to the rejection."
    return f"- **Decision Driver:** AI Model Analysis\n- **Key Strengths:** {strengths}\n- **Major Concerns:** {concerns}\n- **Summary:** {summary}"

def apply_business_rules(applicant_data):
    if applicant_data.get('age', 25) < 21: return {"decision": "Reject", "reason": "Rule: Applicant must be at least 21."}
    if applicant_data.get('avg_monthly_earnings', 0) < 12000: return {"decision": "Reject", "reason": "Rule: Minimum monthly income not met."}
    if applicant_data.get('loan_to_income_ratio', 100) > 4.0: return {"decision": "Reject", "reason": "Rule: Loan-to-income ratio is too high."}
    return None

def make_loan_decision(name, age, gender, city, education, dependent_count, location_stability,
                       avg_monthly_earnings, recurring_expenses, savings, assets_inr, credit_score, 
                       credit_history_length_months, credit_card_user,
                       platform_primary, platform_secondary, sector, rating, tenure_platform_1_months, 
                       tenure_platform_2_months, working_hours_per_day,
                       loan_amount_requested, loan_term_months, purpose_of_loan, num_loan_rejections_6mo):
    
    # --- FIX: Simplified return values for error cases ---
    if not pipeline or not interest_pipeline:
        return "Error", "Models not loaded. Check server logs.", "", "N/A", None

    required_fields = [name, age, gender, city, education, dependent_count, location_stability, avg_monthly_earnings,
                       recurring_expenses, savings, assets_inr, credit_card_user,
                       platform_primary, sector, rating, tenure_platform_1_months,
                       working_hours_per_day, loan_amount_requested, loan_term_months, purpose_of_loan,
                       num_loan_rejections_6mo]
                       
    if any(field is None or str(field) == '' for field in required_fields):
        return "Error", "Please fill in all required fields.", "", "N/A", None
    
    # --- The rest of the function logic remains the same ---
    applicant_dict = {
        'name': name, 'age': age, 'gender': gender, 'education': education, 'dependent_count': dependent_count,
        'location_stability': location_stability, 'avg_monthly_earnings': avg_monthly_earnings,
        'recurring_expenses': recurring_expenses, 'savings': savings, 'assets_inr': assets_inr, 
        'credit_score': credit_score or 300, 'credit_history_length_months': credit_history_length_months or 0,
        'credit_card_user': credit_card_user, 'platform_primary': platform_primary, 
        'platform_secondary': platform_secondary or "None", 'sector': sector, 'rating': rating,
        'tenure_platform_1_months': tenure_platform_1_months,
        'tenure_platform_2_months': tenure_platform_2_months or 0, 'working_hours_per_day': working_hours_per_day,
        'loan_amount_requested': loan_amount_requested, 'loan_term_months': loan_term_months,
        'purpose_of_loan': purpose_of_loan, 'city': city, 'num_loan_rejections_6mo': num_loan_rejections_6mo,
        'earnings_trend_slope_6mo': 0.0
    }
    applicant_df = pd.DataFrame([applicant_dict])
    applicant_df_eng = engineer_features(applicant_df)
    
    final_decision, reason, probability = "Error", "Internal Error", 0.0
    
    rule_result = apply_business_rules(applicant_df_eng.iloc[0].to_dict())
    
    if rule_result:
        final_decision, reason = rule_result['decision'], rule_result['reason']
    else:
        probability = pipeline.predict_proba(applicant_df_eng)[:, 1][0]
        decision_by_ai = "Approve" if probability >= 0.5 else "Reject"
        applicant_transformed = pipeline.named_steps['preprocessor'].transform(applicant_df_eng)
        shap_vals = explainer.shap_values(applicant_transformed)[0]
        reason = get_narrative_reasoning(shap_vals, TRANSFORMED_FEATURE_NAMES, decision_by_ai)
        if decision_by_ai == "Approve":
            final_decision = "Approve"
            predicted_rate = interest_pipeline.predict(applicant_df_eng)[0]
            interest_rate = round(np.clip(predicted_rate, 12.0, 30.0), 2)
            sanction_factor = np.interp(probability, [0.5, 1.0], [0.6, 1.0])
            sanctioned_amount = min(round(loan_amount_requested * sanction_factor, -3), loan_amount_requested)
            loan_term = int(loan_term_months)
            emi = calculate_emi(sanctioned_amount, interest_rate, loan_term)
            disposable_income = applicant_df_eng.iloc[0]['disposable_income']
            max_affordable_emi = disposable_income * 0.60
            if emi > max_affordable_emi:
                final_decision = "Reject"
                reason = f"**Affordability Check Failed:** Est. EMI ₹{int(emi):,} > Max Affordable ₹{int(max_affordable_emi):,}."
        else:
            final_decision = "Reject"

    if final_decision == "Approve":
        emi = calculate_emi(sanctioned_amount, interest_rate, loan_term)
        details_md = f"**Final Sanctioned Amount:** ₹{int(sanctioned_amount):,}\n**Data-Driven Interest Rate:** {interest_rate:.2f}% p.a.\n**Loan Term:** {loan_term} months"
        emi_text = f"₹{emi:,.2f} per month"
    else:
        details_md = ""
        emi_text = "N/A"
        
    plt.figure()
    applicant_transformed = pipeline.named_steps['preprocessor'].transform(applicant_df_eng)
    shap_vals = explainer.shap_values(applicant_transformed)[0]
    shap.plots.waterfall(shap.Explanation(values=shap_vals, base_values=explainer.expected_value, data=applicant_transformed[0], feature_names=TRANSFORMED_FEATURE_NAMES), show=False)
    plt.tight_layout()
    fig = plt.gcf()
        
    # --- FIX: Return the simple text decision for the label ---
    return final_decision, reason, details_md, emi_text, fig

# --- 3. CREATE THE FINAL GRADIO INTERFACE ---

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=fonts.GoogleFont("Inter"),
    text_size=gr.themes.sizes.text_md
)

with gr.Blocks(theme=theme, title="AI Loan Officer") as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 20px;'>AI Loan Officer for the Gig Economy</h1>")
    gr.Markdown("<h3 style='text-align: center; color: #666;'>Enter applicant details to receive a data-driven loan decision.</h3>")
    
    with gr.Tabs():
        with gr.TabItem("👤 Personal & Work"):
            with gr.Group():
                gr.Markdown("#### **Applicant's Basic Information**")
                with gr.Row():
                    name = gr.Textbox(label="Full Name", placeholder="e.g., Priya Singh")
                    age = gr.Slider(minimum=18, maximum=70, value=None, step=1, label="Age")
                with gr.Row():
                    gender = gr.Radio(choices=["Female", "Male", "Other"], label="Gender", value=None)
                    city = gr.Dropdown(label="City", choices=["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Gurugram", "Noida", "Other"], allow_custom_value=True, value=None)
                with gr.Row():
                    education = gr.Dropdown(choices=["10th", "12th", "Graduate", "Post-graduate", "Other"], label="Highest Education", value=None)
                    dependent_count = gr.Slider(minimum=0, maximum=10, value=None, step=1, label="Number of Dependents")
                location_stability = gr.Dropdown(choices=["stable", "semi-stable", "unstable"], label="Location Stability", value=None)
            
            with gr.Group():
                gr.Markdown("#### **Gig Work Profile**")
                with gr.Row():
                    platform_primary = gr.Dropdown(choices=["Uber", "Ola", "Zomato", "Swiggy", "UrbanClap", "Other"], label="Primary Gig Platform", value=None)
                    platform_secondary = gr.Textbox(label="Secondary Platform (if any)", placeholder="e.g., Dunzo")
                with gr.Row():
                    sector = gr.Dropdown(choices=["cab", "food_delivery", "e-commerce", "domestic", "beauty", "other"], label="Primary Sector", value=None)
                    rating = gr.Slider(minimum=1.0, maximum=5.0, step=0.1, value=None, label="Platform Rating (1-5)")
                with gr.Row():
                    tenure_platform_1_months = gr.Slider(minimum=0, maximum=120, value=None, step=1, label="Months on Primary Platform")
                    tenure_platform_2_months = gr.Slider(minimum=0, maximum=120, value=None, step=1, label="Months on Secondary Platform")

        with gr.TabItem("💰 Financial Details"):
            with gr.Group():
                gr.Markdown("#### **Income & Expenses**")
                with gr.Row():
                    avg_monthly_earnings = gr.Slider(minimum=10000, maximum=150000, value=None, step=1000, label="Average Monthly Earnings (₹)")
                    recurring_expenses = gr.Slider(minimum=0, maximum=100000, value=None, step=1000, label="Recurring Monthly Expenses (₹)")
                with gr.Row():
                    savings = gr.Number(label="Total Savings (₹)", value=None)
                    assets_inr = gr.Number(label="Total Assets Value (Vehicle, etc.) (₹)", value=None, info="Enter the approximate total value of significant assets.")
            with gr.Group():
                gr.Markdown("#### **Credit Information**")
                credit_card_user = gr.Radio(choices=["Yes", "No"], label="Is the applicant a credit card user?", value=None)
                with gr.Row(visible=False) as credit_details_group:
                    credit_score = gr.Slider(minimum=300, maximum=850, value=None, step=10, label="Credit Score (e.g., CIBIL)")
                    credit_history_length_months = gr.Slider(minimum=0, maximum=120, value=None, step=1, label="Credit History (in Months)")
                def show_credit_details(is_user): return gr.update(visible=(is_user == "Yes"))
                credit_card_user.change(fn=show_credit_details, inputs=credit_card_user, outputs=credit_details_group)
            working_hours_per_day = gr.Slider(minimum=1, maximum=16, value=None, step=1, label="Average Daily Hours")

        with gr.TabItem("🏦 Loan Request"):
            with gr.Group():
                loan_amount_requested = gr.Slider(minimum=10000, maximum=500000, value=None, step=5000, label="Loan Amount Requested (₹)")
                loan_term_months = gr.Slider(minimum=6, maximum=60, value=None, step=6, label="Desired Loan Term (Months)")
                purpose_of_loan = gr.Dropdown(choices=["vehicle", "business", "personal", "home_improvement", "debt_consolidation"], label="Purpose of Loan", value=None)
                num_loan_rejections_6mo = gr.Slider(minimum=0, maximum=10, value=None, step=1, label="Loan Rejections in Last 6 Months")

    gr.Markdown("---")
    submit_button = gr.Button("Fill All Fields to Evaluate", variant="primary", size="lg", interactive=False)

    with gr.Accordion("✅ **Decision & Analysis**", open=False) as output_accordion:
        decision_label = gr.Label(label="Final Decision", scale=1)
        reasoning_output = gr.Markdown(label="Decision Reasoning")
        with gr.Row():
            details_output = gr.Markdown(label="Loan Terms")
            emi_output = gr.Textbox(label="Estimated Monthly Payment (EMI)", interactive=False)
        shap_plot_output = gr.Plot(label="Key Factors Influencing the Decision (SHAP Analysis)")
        
    def submit_and_open_accordion(*args):
        # A simple check for NoneType before calling the main function
        label, reason, details, emi, plot = make_loan_decision(*args)
        return label, reason, details, emi, plot, gr.Accordion(open=True)

    all_inputs = [
        name, age, gender, city, education, dependent_count, location_stability,
        avg_monthly_earnings, recurring_expenses, savings, assets_inr, credit_score,
        credit_history_length_months, credit_card_user,
        platform_primary, platform_secondary, sector, rating, tenure_platform_1_months,
        tenure_platform_2_months, working_hours_per_day,
        loan_amount_requested, loan_term_months, purpose_of_loan, num_loan_rejections_6mo
    ]
    
    # This list defines which inputs must be filled for the button to activate
    required_inputs = [
        name, age, gender, city, education, dependent_count, location_stability,
        avg_monthly_earnings, recurring_expenses, savings, assets_inr, credit_card_user,
        platform_primary, sector, rating, tenure_platform_1_months,
        working_hours_per_day, loan_amount_requested, loan_term_months, purpose_of_loan,
        num_loan_rejections_6mo
    ]

    def update_submit_button_state(*input_values):
        if any(val is None or (isinstance(val, str) and not val.strip()) for val in input_values):
            return gr.Button(value="Fill All Required Fields", interactive=False)
        return gr.Button(value="Evaluate Loan Application", interactive=True)
    
    for component in required_inputs:
        component.change(fn=update_submit_button_state, inputs=required_inputs, outputs=submit_button)
    
    submit_button.click(
        fn=submit_and_open_accordion,
        inputs=all_inputs,
        outputs=[decision_label, reasoning_output, details_output, emi_output, shap_plot_output, output_accordion]
    )

if __name__ == "__main__":
    demo.launch()