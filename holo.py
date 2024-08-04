from flask import Flask, request, jsonify
from openai import OpenAI
from openai import AzureOpenAI
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

# Initialize Flask application
app = Flask(__name__)

# Set up logging
#logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Initialize OpenAI client
#client = OpenAI(api_key='----')
client = AzureOpenAI(
    api_key="----",
    api_version="2024-02-01",
    azure_endpoint="https://holo.openai.azure.com/"
)

deployment_name = "gpt-35-turbo-instruct"

def get_prompt_embedding(prompt_text):
    return np.array(client.embeddings.create(input=[prompt_text], model="text-embedding-3-small").data[0].embedding)

def find_best_components(prompt_embedding, top_n=5):
    df = pd.read_csv('embedded_components.csv')
    df['ada_embedding'] = df['ada_embedding'].apply(eval).apply(np.array)
    similarities = cosine_similarity([prompt_embedding], df['ada_embedding'].tolist())[0]
    top_indices = np.argsort(similarities)[-top_n:]
    return df.iloc[top_indices]

def generate_layout_config(prompt_text):
    prompt_embedding = get_prompt_embedding(prompt_text)
    top_components = find_best_components(prompt_embedding)
    layout_config = {
        "topSection": {
            "customerInfo": {
                "components": ["Name", "CustomerID", "Status", "RelationshipManager"]
            }
        },
        "mainContentArea": {
            "sections": [
                {"name": row['component'], "cols": row['cols']} for index, row in top_components.iterrows()
            ]
        }
    }
    return layout_config

def generate_layout(prompt_text, layout_config):
    system_message = f"""
    You are HOLO, a highly advanced AI designed to dynamically generate UI layouts for a bank.
    Your goal is to create the most efficient and user-friendly Customer 360 views for Digital Branch, a banking app for bank employees for different roles.
    You should generate a layout with at least eight widgets in the mainContentArea that displays the most relevant information for a given customer based on the customer data given.

    You need to reply in JSON format with the following properties:
    {{
      topSection: {{
        customerInfo: an array of components used for customer information based on the customer data and role
      }},
      mainContentArea: {{
        sections: an array of relevant components displayed in the main content area based on the customer data and role
      }}
    }}

    The possible values for the mainContentArea sections are:
    AccountOverview, HighValueTransactions, PersonalizedOffers, RelationshipDetails, ESGImpactOverview, InvestmentESGAnalysis, RecentTransactions,
    LoanDetails,CreditCardSummary, SavingsGoals, BudgetOverview, CustomerInsights, SpendingTrends, FinancialGoals, RetirementPlanning, InsurancePolicies,
    TaxSummary, MortgageDetails, InvestmentPortfolio, CharitableDonations, EmergencyFundStatus, IncomeBreakdown, CreditScore, DebtToIncomeRatio, CreditScoreHistory

    The possible values for the topSection are:
    Account Summary, Transaction History, Financial Health Score, Interaction History, Product Usage Overview, Behavioral Insights, Loan and Credit Status,
    Investment Portfolio, Customer Preferences, Engagement Metrics, Customer Sentiment Analysis, Risk and Compliance Dashboard, ESG Metrics, Notifications and Alerts,
    Customer Feedback, Customer Profile Summary, Spending Analysis, Savings Goals Tracker, Income Analysis, Goal Setting and Tracking, Budgeting Tools,
    Loan Eligibility Calculator, Credit Score Monitor, Mortgage Calculator, Net Worth Tracker, Investment Performance Tracker, Alerts and Notifications,
    Tax Reporting Tools, Insurance Policy Overview, Behavioral Analysis and Recommendations, Predictive Analytics Dashboard.

    For example, the response should be:
    {{
      "topSection": {{
        "customerInfo": {{
          "components": ["name", "email", "phone"]
        }}
      }},
      "mainContentArea": {{
        "sections": [
          {{
            "name": "relevent widget",
            "cols": "auto"
          }}
        ] 
      }}
    }}
    
    """

    prompt = f"""
      {prompt_text}.
    """
    #prompt = f"""
    #  {prompt_text}.
    #  The recommended layout configuration is {layout_config}.
    #"""
    print(prompt)

   #response = client.chat.completions.create(
   #     model="gpt-4o-mini",
   #     messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
   #     max_tokens=250,
   #     n=1,
   #     stop=None,
   #     temperature=0.7,
   #)

   #enhanced_layout = response.choices[0].message.content

    response = client.completions.create(
        model=deployment_name,
        prompt=f"{system_message}\n\n{prompt_text}",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        best_of=1,
        stop=None
    )
    
    enhanced_layout = response.choices[0].text
    # Parse the JSON layout from the response
    try:
      layout_json = json.loads(enhanced_layout.strip('```json\n').strip('```'))
    except json.JSONDecodeError:
      layout_json = {"error": "Failed to parse JSON from response"}
    app.logger.debug(f"\n--- Holo's Generated UI ---\n{layout_json}\n--------------------------------------------\n")

    return layout_json

def determine_customer_segment(prompt_text):
    system_message = f"""
    You are HOLO, a highly advanced AI designed to dynamically generate UI layouts for a bank.
    Your goal is to determine the customer segment based on the given customer information.

    You need to reply with the customer segments based on the given information and a summary of the customer's financial status and needs.
    A customer can have multiple segments based on their profile and needs.

    The possible values for the customer segment but not limited to are:
    Young Professional, Family, Senior Citizen, Small Business Owner, High-Net-Worth Individual, Middle Income, Student, Freelancer/Gig Worker, New Immigrant, 
    Non-Profit Organization, Retiree, Traveler, Young Couple, Investor, Seasonal Worker, Tech-Savvy Customer, Frequent Flyer, Remote Worker,
    Military Personnel, Entrepreneur, and Healthcare Professional.
    """

    prompt = f"""
      {prompt_text}.
    """
    #print(prompt)

    #response = client.chat.completions.create(
    #    model="gpt-4o-mini",
    #    messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
    #    max_tokens=1000,
    #    n=1,
    #    stop=None,
    #    temperature=0.7,
    #)
    
    #customer_segment = response.choices[0].message.content

    response = client.completions.create(
        model=deployment_name,
        prompt=f"{system_message}\n\n{prompt_text}",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        best_of=1,
        stop=None
    )
    customer_segment = response.choices[0].text
    app.logger.debug(f"\n--- Holo's Generated Customer Summary---\n{customer_segment}\n--------------------------------------------\n")


    return customer_segment

@app.route('/api/v1/generate-ui', methods=['POST'])
def generate_ui():
    data = request.json
    customer_info = data.get('customerInfo')
    role = data.get('role')
    task = data.get('task')

    prompt = f"""
    Create a layout for a Customer 360 view for a { role } viewing the details of a customer with this info: { customer_info }.
    The task of the { role } is { task}
    """

    app.logger.debug(f"\n--- Generated Prompt for Customer 360 View ---\n{prompt}\n--------------------------------------------\n")

    #layout_config = generate_layout_config(prompt)
    layout_config = ""
    generated_layout = generate_layout(prompt, layout_config)
    
    return jsonify({"layout": generated_layout})

@app.route('/api/v1/customer-segment', methods=['POST'])
def get_customer_segment():
    data = request.json
    customer_info = data.get('customerInfo')
    
    prompt = f"""
    Determine the customer segment for this customer: { customer_info }.
    """
    #app.logger.debug(f'Generated prompt: {prompt}')
    customer_segment = determine_customer_segment(prompt)
    
    return jsonify({"customer_segment": customer_segment})

if __name__ == '__main__':
    app.run(debug=True, port=3033)

