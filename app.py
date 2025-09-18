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
    """
    df_eng = df.copy()
    
    for col in ['avg_monthly_earnings', 'loan_amount_requested', 'savings', 'recurring_expenses', 
                'tenure_platform_1_months', 'tenure_platform_2_months', 'rating', 'working_hours_per_day']:
        if col in df_eng.columns:
            df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)

    capped_savings = df_eng['savings'].clip(upper=200000)
    df_eng['savings_log_capped'] = np.log1p(capped_savings)
    
    df_eng['avg_monthly_earnings'] = df_eng['avg_monthly_earnings'].clip(lower=1)
    df_eng['loan_to_income_ratio'] = df_eng['loan_amount_requested'] / df_eng['avg_monthly_earnings']
    
    df_eng['total_tenure_months'] = df_eng['tenure_platform_1_months'] + df_eng['tenure_platform_2_months']
    
    df_eng['disposable_income'] = df_eng['avg_monthly_earnings'] - df_eng['recurring_expenses']
    
    log_tenure = np.log1p(df_eng['total_tenure_months'])
    log_avg_earnings = np.log1p(df_eng['avg_monthly_earnings'])
    df_eng['gig_performance_score'] = (
        df_eng['rating'] * log_tenure * log_avg_earnings * df_eng['working_hours_per_day']
    ).fillna(0)
    
    return df_eng

def calculate_emi(principal, annual_rate, term_months):
    if principal <= 0 or annual_rate <= 0 or term_months <= 0: return 0
    monthly_rate = (annual_rate / 100) / 12
    emi = principal * monthly_rate * (1 + monthly_rate)**term_months / ((1 + monthly_rate)**term_months - 1)
    return emi

def get_narrative_reasoning(shap_values, feature_names, decision, applicant_data, top_n=3):
    """Generates a conversational, human-like loan officer's assessment."""
    applicant_name = applicant_data.get('name', 'Applicant').split(" ")[0]
    sv_series = pd.Series(shap_values, index=feature_names)

    narrative_map = {
        'gig_performance_score': {
            'positive': "Your strong Gig Performance Score, which considers your platform rating, tenure, and earnings, was a key factor. It indicates you are a consistent and reliable professional in the gig economy.",
            'negative': "The Gig Performance Score was a point of concern. This score looks at platform rating, tenure, and earnings, and a lower value suggests some risk or inconsistency in your work profile."
        },
        'disposable_income': {
            'positive': f"We were impressed with your healthy disposable income of ₹{int(applicant_data.get('disposable_income', 0)):,}. This demonstrates a strong capacity to comfortably manage monthly loan payments.",
            'negative': f"After reviewing your income and expenses, your disposable income of ₹{int(applicant_data.get('disposable_income', 0)):,} was a concern. This narrow margin could make it difficult to take on new debt."
        },
        'loan_to_income_ratio': {
            'positive': "The loan amount requested is very reasonable compared to your earnings, resulting in a good Loan-to-Income ratio. This was a significant positive point in your favor.",
            'negative': "The primary concern was the high Loan-to-Income ratio. The requested loan amount is quite large relative to your income, which raises questions about long-term affordability."
        },
        'savings_log_capped': {
            'positive': "Your savings history is commendable and shows excellent financial discipline. This financial cushion significantly reduces risk and was viewed very positively.",
            'negative': "The level of savings was lower than anticipated, which can indicate limited capacity to handle unexpected financial shocks during the loan term."
        },
        'credit_score': {
            'positive': f"Your credit score of {int(applicant_data.get('credit_score', 0))} is solid. It reflects a responsible credit history and played an important role in this decision.",
            'negative': f"The credit score of {int(applicant_data.get('credit_score', 0))} was a contributing factor. A lower score can suggest past difficulties with credit obligations."
        },
        'total_tenure_months': {
            'positive': f"With a total of {int(applicant_data.get('total_tenure_months', 0))} months of experience on gig platforms, you have demonstrated stability and a long-term presence in your field. This consistency is a major strength.",
            'negative': f"Your relatively short time on the gig platforms ({int(applicant_data.get('total_tenure_months', 0))} months) means we have less data to establish a long-term trend of stability."
        },
        'num_loan_rejections_6mo': {
            'positive': "We noted that you have not had recent loan rejections, which is a positive signal.",
            'negative': f"The {int(applicant_data.get('num_loan_rejections_6mo', 0))} loan rejections in the past six months were a significant red flag, suggesting other lenders may have also identified risks in your profile."
        }
    }

    pos_contributors_tech = sv_series[sv_series > 0].nlargest(top_n)
    neg_contributors_tech = sv_series[sv_series < 0].nsmallest(top_n)

    def get_narrative_for_feature(tech_name):
        clean_name = tech_name.replace('num__', '').replace('cat__', '')
        for key in narrative_map:
            if clean_name.startswith(key):
                return narrative_map[key]
        return None

    positive_narratives = [p['positive'] for feature, _ in pos_contributors_tech.items() if (p := get_narrative_for_feature(feature))]
    negative_narratives = [n['negative'] for feature, _ in neg_contributors_tech.items() if (n := get_narrative_for_feature(feature))]

    if decision == "Approve":
        report = f"### Loan Assessment for {applicant_name}\n\n"
        report += f"**Dear {applicant_name},**\n\n"
        report += "We are pleased to inform you that your loan application has been **approved**. Our decision was based on a comprehensive review of your profile, with several key strengths noted by our AI model.\n\n"
        if positive_narratives:
            report += "#### Key Positive Factors:\n"
            for p in positive_narratives: report += f"- {p}\n"
        if negative_narratives:
            report += "\n#### Areas Considered:\n"
            report += "We did consider some minor risk factors, but your overall strong profile outweighed them. For instance:\n"
            for n in negative_narratives: report += f"- {n.replace('was a concern', 'was noted').replace('was a point of concern', 'was reviewed')}\n"
        report += "\nCongratulations! We believe in supporting the ambitions of gig economy professionals like you."
    else:
        report = f"### Loan Assessment for {applicant_name}\n\n"
        report += f"**Dear {applicant_name},**\n\n"
        report += "Thank you for your interest. After a careful review of your application, we regret to inform you that we are unable to approve your loan request at this time. Our AI model identified a few key areas of concern that led to this decision.\n\n"
        if negative_narratives:
            report += "#### Primary Reasons for Decision:\n"
            for n in negative_narratives: report += f"- {n}\n"
        if positive_narratives:
            report += "\n#### Profile Strengths Acknowledged:\n"
            report += "We want to acknowledge the strong points in your profile, which we did take into consideration:\n"
            for p in positive_narratives: report += f"- {p}\n"
        report += "\nWe encourage you to focus on strengthening the areas mentioned above. We wish you the best and invite you to reapply in the future should your circumstances change."
    return report


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
    
    if not pipeline or not interest_pipeline:
        return "Error", "Models not loaded. Check server logs.", "", "N/A", None

    required_fields = [name, age, gender, city, education, dependent_count, location_stability, avg_monthly_earnings,
                       recurring_expenses, savings, assets_inr, credit_card_user,
                       platform_primary, sector, rating, tenure_platform_1_months,
                       working_hours_per_day, loan_amount_requested, loan_term_months, purpose_of_loan,
                       num_loan_rejections_6mo]
                       
    if any(field is None or str(field) == '' for field in required_fields):
        return "Error", "Please fill in all required fields.", "", "N/A", None
    
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
        
        reason = get_narrative_reasoning(shap_vals, TRANSFORMED_FEATURE_NAMES, decision_by_ai, applicant_df_eng.iloc[0].to_dict())

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
                reason = f"**Affordability Check Failed:** Estimated EMI of ₹{int(emi):,} exceeds the maximum affordable amount of ₹{int(max_affordable_emi):,} based on your disposable income."
        else:
            final_decision = "Reject"

    if final_decision == "Approve":
        predicted_rate = interest_pipeline.predict(applicant_df_eng)[0]
        interest_rate = round(np.clip(predicted_rate, 12.0, 30.0), 2)
        sanction_factor = np.interp(probability, [0.5, 1.0], [0.6, 1.0])
        sanctioned_amount = min(round(loan_amount_requested * sanction_factor, -3), loan_amount_requested)
        loan_term = int(loan_term_months)
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
    gr.Markdown("<h3 style='text-align: center; color: #666;'>A sample case is pre-filled. Click 'Evaluate' or modify the details below.</h3>")
    
    with gr.Tabs():
        with gr.TabItem("👤 Personal & Work"):
            with gr.Group():
                gr.Markdown("#### **Applicant's Basic Information**")
                with gr.Row():
                    name = gr.Textbox(label="Full Name", value="Rohan Sharma") # Default value
                    age = gr.Slider(minimum=18, maximum=70, value=32, step=1, label="Age") # Default value
                with gr.Row():
                    gender = gr.Radio(choices=["Female", "Male", "Other"], label="Gender", value="Male") # Default value
                    city = gr.Dropdown(label="City", choices=["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Gurugram", "Noida", "Other"], allow_custom_value=True, value="Gurugram") # Default value
                with gr.Row():
                    education = gr.Dropdown(choices=["10th", "12th", "Graduate", "Post-graduate", "Other"], label="Highest Education", value="12th") # Default value
                    dependent_count = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Number of Dependents") # Default value
                location_stability = gr.Dropdown(choices=["stable", "semi-stable", "unstable"], label="Location Stability", value="stable") # Default value
            
            with gr.Group():
                gr.Markdown("#### **Gig Work Profile**")
                with gr.Row():
                    platform_primary = gr.Dropdown(choices=["Uber", "Ola", "Zomato", "Swiggy", "UrbanClap", "Other"], label="Primary Gig Platform", value="Uber") # Default value
                    platform_secondary = gr.Textbox(label="Secondary Platform (if any)", placeholder="e.g., Dunzo", value="Zomato") # Default value
                with gr.Row():
                    sector = gr.Dropdown(choices=["cab", "food_delivery", "e-commerce", "domestic", "beauty", "other"], label="Primary Sector", value="cab") # Default value
                    rating = gr.Slider(minimum=1.0, maximum=5.0, step=0.1, value=4.7, label="Platform Rating (1-5)") # Default value
                with gr.Row():
                    tenure_platform_1_months = gr.Slider(minimum=0, maximum=120, value=30, step=1, label="Months on Primary Platform") # Default value
                    tenure_platform_2_months = gr.Slider(minimum=0, maximum=120, value=12, step=1, label="Months on Secondary Platform") # Default value

        with gr.TabItem("💰 Financial Details"):
            with gr.Group():
                gr.Markdown("#### **Income & Expenses**")
                with gr.Row():
                    avg_monthly_earnings = gr.Slider(minimum=10000, maximum=150000, value=45000, step=1000, label="Average Monthly Earnings (₹)") # Default value
                    recurring_expenses = gr.Slider(minimum=0, maximum=100000, value=25000, step=1000, label="Recurring Monthly Expenses (₹)") # Default value
                with gr.Row():
                    savings = gr.Number(label="Total Savings (₹)", value=85000) # Default value
                    assets_inr = gr.Number(label="Total Assets Value (Vehicle, etc.) (₹)", value=350000, info="Enter the approximate total value of significant assets.") # Default value
            with gr.Group():
                gr.Markdown("#### **Credit Information**")
                credit_card_user = gr.Radio(choices=["Yes", "No"], label="Is the applicant a credit card user?", value="Yes") # Default value
                with gr.Row(visible=True) as credit_details_group: # Default visible=True
                    credit_score = gr.Slider(minimum=300, maximum=850, value=740, step=10, label="Credit Score (e.g., CIBIL)") # Default value
                    credit_history_length_months = gr.Slider(minimum=0, maximum=120, value=36, step=1, label="Credit History (in Months)") # Default value
                def show_credit_details(is_user): return gr.update(visible=(is_user == "Yes"))
                credit_card_user.change(fn=show_credit_details, inputs=credit_card_user, outputs=credit_details_group)
            working_hours_per_day = gr.Slider(minimum=1, maximum=16, value=10, step=1, label="Average Daily Hours") # Default value

        with gr.TabItem("🏦 Loan Request"):
            with gr.Group():
                loan_amount_requested = gr.Slider(minimum=10000, maximum=500000, value=150000, step=5000, label="Loan Amount Requested (₹)") # Default value
                loan_term_months = gr.Slider(minimum=6, maximum=60, value=36, step=6, label="Desired Loan Term (Months)") # Default value
                purpose_of_loan = gr.Dropdown(choices=["vehicle", "business", "personal", "home_improvement", "debt_consolidation"], label="Purpose of Loan", value="vehicle") # Default value
                num_loan_rejections_6mo = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Loan Rejections in Last 6 Months") # Default value

    gr.Markdown("---")
    submit_button = gr.Button("Evaluate Loan Application", variant="primary", size="lg", interactive=True) # Default interactive=True

    with gr.Accordion("✅ **Decision & Analysis**", open=False) as output_accordion:
        decision_label = gr.Label(label="Final Decision", scale=1)
        reasoning_output = gr.Markdown(label="Decision Reasoning")
        with gr.Row():
            details_output = gr.Markdown(label="Loan Terms")
            emi_output = gr.Textbox(label="Estimated Monthly Payment (EMI)", interactive=False)
        shap_plot_output = gr.Plot(label="Key Factors Influencing the Decision (SHAP Analysis)")
        
    def submit_and_open_accordion(*args):
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

