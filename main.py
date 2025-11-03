from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import datetime
from budget_calc import calculate_advanced_budget_recommendations
from financial_analyzer import calculate_financial_health
from goal_planner import calculate_goal_feasibility, track_goal_progress
from credit_advisor import analyze_credit_and_debt
from sample_profiles import SAMPLE_PROFILES
from investment_advisor import generate_investment_guidance

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Needed for sessions

@app.route("/load_sample/<profile_id>")
def load_sample(profile_id):
    if profile_id in SAMPLE_PROFILES:
        return jsonify(SAMPLE_PROFILES[profile_id])
    return jsonify({"error": "Profile not found"}), 404


def get_current_year():
    return datetime.datetime.now().year


@app.route("/", methods=['GET', 'POST'])
@app.route("/index.html", methods=['GET', 'POST'])
def index():
    year = get_current_year()

    if request.method == 'POST':
        # Get form data
        monthly_income = float(request.form.get('monthly_income', 0))
        age_range = request.form.get('age_range', '')
        family_size = request.form.get('family_size', '')
        location = request.form.get('location', '')
        
        # Get expense data
        expenses = {
            'housing': float(request.form.get('housing_expense', 0)),
            'transportation': float(request.form.get('transportation_expense', 0)),
            'food': float(request.form.get('food_expense', 0)),
            'utilities': float(request.form.get('utilities_expense', 0)),
            'healthcare': float(request.form.get('healthcare_expense', 0)),
            'debt_payments': float(request.form.get('debt_expense', 0)),
            'discretionary': float(request.form.get('discretionary_expense', 0))
        }

        # Calculate financial health
        financial_health = calculate_financial_health(monthly_income, expenses)
        
        # Store form data in session
        session['budget_data'] = {
            'monthly_income': monthly_income,
            'age_range': age_range,
            'family_size': family_size,
            'location': location,
            'expenses': expenses,
            'financial_health': financial_health
        }

        # Calculate budget recommendations and store in session
        recommendations = calculate_advanced_budget_recommendations(
            monthly_income, age_range, family_size, location
        )
        session['recommendations'] = recommendations

        # Redirect to generic.html after form submission
        return redirect(url_for('generic'))

    # For GET requests, just render the index page
    return render_template("index.html", year=year)


@app.route("/generic", methods=['GET'])
def generic():
    year = get_current_year()

    # Retrieve data from session
    budget_data = session.get('budget_data', {})
    recommendations = session.get('recommendations', {})

    # Prepare chart data for the pie chart
    chart_data = None
    if recommendations:
        chart_data = {
            'labels': list(recommendations.keys()),
            'percentages': [rec['percentage'] for rec in recommendations.values()],
            'amounts': [rec['amount'] for rec in recommendations.values()],
            'colors': [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40',
                '#FF6384', '#C9CBCF', '#4BC0C0', '#FFCD56', '#36A2EB', '#FF6384'
            ]
        }

    return render_template(
        "generic.html",
        year=year,
        recommendations=recommendations,
        budget_data=budget_data,
        chart_data=chart_data
    )


@app.route("/goals", methods=['GET', 'POST'])
def goals():
    year = get_current_year()
    
    if request.method == 'POST':
        # Clear all goals if requested
        if request.form.get('action') == 'clear':
            session['goals'] = []
            return redirect(url_for('goals'))

        # Get goal data
        goal_data = {
            'name': request.form.get('goal_name'),
            'target_amount': float(request.form.get('target_amount')),
            'current_amount': float(request.form.get('current_amount', 0)),
            'target_date': datetime.datetime.strptime(request.form.get('target_date'), '%Y-%m-%d') if request.form.get('target_date') else None
        }
        
        # Get financial data from session
        budget_data = session.get('budget_data', {})
        monthly_income = budget_data.get('monthly_income', 0)
        expenses = budget_data.get('expenses', {})
        monthly_expenses = sum(expenses.values()) if expenses else 0
        
        # Calculate goal feasibility
        feasibility = calculate_goal_feasibility(
            goal_data['target_amount'],
            goal_data['current_amount'],
            monthly_income,
            monthly_expenses,
            goal_data['target_date']
        )
        
        # Calculate progress
        progress = track_goal_progress(
            goal_data['target_amount'],
            goal_data['current_amount'],
            datetime.datetime.now(),
            goal_data['target_date']
        )
        
        # Store goal in session
        session['goals'] = session.get('goals', [])
        goal_data.update({
            'feasibility': feasibility,
            'progress': progress
        })
        session['goals'].append(goal_data)
        
        return redirect(url_for('goals'))
    
    return render_template(
        "goals.html",
        year=year,
        goals=session.get('goals', []),
        budget_data=session.get('budget_data', {})
    )

@app.route("/elements")
def elements():
    year = get_current_year()
    return render_template("elements.html", year=year)

@app.route("/credit", methods=['GET', 'POST'])
def credit():
    year = get_current_year()
    
    if request.method == 'POST':
        # Get credit and debt inputs
        total_debt = float(request.form.get('total_debt', 0))
        monthly_debt_payment = float(request.form.get('monthly_debt_payment', 0))
        avg_interest_rate = float(request.form.get('avg_interest_rate', 0))
        credit_utilization_pct = float(request.form.get('credit_utilization_pct', 0))
        on_time_payment_pct = float(request.form.get('on_time_payment_pct', 0))
        open_credit_lines = int(request.form.get('open_credit_lines', 0))
        hard_inquiries_12m = int(request.form.get('hard_inquiries_12m', 0))

        # Analyze credit and debt
        analysis = analyze_credit_and_debt(
            total_debt,
            monthly_debt_payment,
            avg_interest_rate,
            credit_utilization_pct,
            on_time_payment_pct,
            open_credit_lines,
            hard_inquiries_12m,
        )

        # Store analysis and inputs in session
        session['credit_inputs'] = {
            'total_debt': total_debt,
            'monthly_debt_payment': monthly_debt_payment,
            'avg_interest_rate': avg_interest_rate,
            'credit_utilization_pct': credit_utilization_pct,
            'on_time_payment_pct': on_time_payment_pct,
            'open_credit_lines': open_credit_lines,
            'hard_inquiries_12m': hard_inquiries_12m,
        }
        session['credit_analysis'] = analysis

        return render_template(
            "credit.html",
            year=year,
            analysis=analysis,
            credit_inputs=session.get('credit_inputs')
        )
    
    return render_template("credit.html", year=year, credit_inputs=session.get('credit_inputs'))

@app.route("/download-csv")
def download_csv():
    
    budget_data = session.get('budget_data', {})
    recommendations = session.get('recommendations', {})

    if not recommendations:
        return redirect(url_for('index'))

    
    output = io.StringIO()
    writer = csv.writer(output)

    
    writer.writerow(['Personalized Budget Report'])
    writer.writerow([f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
    writer.writerow([])

    
    writer.writerow(['USER PROFILE'])
    writer.writerow(['Monthly Income', f'₹{budget_data.get("monthly_income", 0):,.0f}'])
    writer.writerow(['Age Range', budget_data.get('age_range', 'Not specified')])
    writer.writerow(['Family Size', budget_data.get('family_size', 'Not specified')])
    writer.writerow(['Location', budget_data.get('location', 'Not specified')])
    writer.writerow([])

    
    writer.writerow(['BUDGET RECOMMENDATIONS'])
    writer.writerow(['Category', 'Percentage', 'Amount (₹)'])

    
    total_amount = 0
    for category, data in recommendations.items():
        writer.writerow([
            category,
            data.get('display_percentage', '0%'),
            data.get('display_amount', '₹0')
        ])
        total_amount += data.get('amount', 0)

    
    writer.writerow(['TOTAL', '100%', f'₹{total_amount:,.0f}'])

    output.seek(0)
    response = Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=budget_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )

    return response

@app.route("/invest", methods=['GET'])
def invest():
    year = get_current_year()
    budget_data = session.get('budget_data', {})
    goals = session.get('goals', [])

    monthly_income = budget_data.get('monthly_income', 0)
    age_range = budget_data.get('age_range', '')
    financial_health = budget_data.get('financial_health')

    guidance = None
    if monthly_income:
        guidance = generate_investment_guidance(
            monthly_income=monthly_income,
            age_range=age_range,
            financial_health=financial_health,
            goals=goals,
        )

    return render_template(
        "investment.html",
        year=year,
        budget_data=budget_data,
        guidance=guidance,
    )

if __name__ == "__main__":
    app.run(debug=True)
