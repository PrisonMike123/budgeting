import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response
import datetime
from budget_calc import calculate_advanced_budget_recommendations
from financial_analyzer import calculate_financial_health
from goal_planner import calculate_goal_feasibility, track_goal_progress
from credit_advisor import analyze_credit_and_debt
from sample_profiles import SAMPLE_PROFILES
from investment_advisor import generate_investment_guidance
import csv
import io
from functools import wraps
from prediction_utils import make_prediction
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['DEBUG'] = True

# Configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-123'),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=1)
)

# Security headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

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
    budget_data = session.get('budget_data', {})
    recommendations = session.get('recommendations', {})
    
    if not budget_data:
        return redirect(url_for('index'))
    
    # Prepare chart data with default values if not present
    expenses = budget_data.get('expenses', {})
    total_expenses = sum(expenses.values()) or 1  # Avoid division by zero
    
    # Generate colors for each category
    colors = [
        '#4e79a7',  # Blue
        '#f28e2b',  # Orange
        '#e15759',  # Red
        '#76b7b2',  # Teal
        '#59a14f',  # Green
        '#edc948',  # Yellow
        '#b07aa1',  # Purple
        '#ff9da7',  # Pink
        '#9c755f',  # Brown
        '#bab0ac'   # Gray
    ]
    
    chart_data = {
        'labels': list(expenses.keys()),
        'data': list(expenses.values()),
        'percentages': [round((v / total_expenses) * 100, 1) for v in expenses.values()],
        'amounts': list(expenses.values()),
        'colors': colors[:len(expenses)],
        'recommended': recommendations.get('category_budgets', {k: 0 for k in expenses})
    }
        
    return render_template("generic.html", 
                         year=year,
                         budget_data=budget_data,
                         recommendations=recommendations,
                         chart_data=chart_data)


@app.route("/update_goal", methods=['POST'])
def update_goal():
    if 'goals' not in session:
        return redirect(url_for('goals'))
        
    try:
        goal_index = int(request.form.get('goal_index'))
        if goal_index < 0 or goal_index >= len(session['goals']):
            raise ValueError("Invalid goal index")
            
        # Update the goal
        session['goals'][goal_index] = {
            'name': request.form.get('goal_name'),
            'target_amount': float(request.form.get('target_amount', 0)),
            'current_amount': float(request.form.get('current_amount', 0)),
            'time_frame': int(request.form.get('time_frame', 12)),
            'created_at': session['goals'][goal_index].get('created_at', datetime.datetime.now().isoformat()),
            'target_date': (datetime.datetime.now() + datetime.timedelta(days=int(request.form.get('time_frame', 12))*30)).isoformat()
        }
        session.modified = True
        
    except (ValueError, IndexError) as e:
        print(f"Error updating goal: {e}")
        
    return redirect(url_for('goals'))

@app.route("/goals", methods=['GET', 'POST'])
def goals():
    year = get_current_year()
    
    if request.method == 'POST':
        # Check if this is a clear action
        if request.form.get('action') == 'clear':
            session.pop('goals', None)
            return redirect(url_for('goals'))
            
        # Get form data for new goal
        goal_name = request.form.get('goal_name')
        target_amount = float(request.form.get('target_amount', 0))
        current_amount = float(request.form.get('current_amount', 0))
        time_frame = int(request.form.get('time_frame', 12))  # in months
        
        # Get user's financial data from session or use defaults
        budget_data = session.get('budget_data', {})
        monthly_income = float(budget_data.get('monthly_income', 5000))
        monthly_savings = float(budget_data.get('financial_health', {}).get('metrics', {}).get('monthly_savings', 1000))
        
        # Calculate target date based on time frame
        target_date = datetime.datetime.now() + datetime.timedelta(days=time_frame*30)
        
        # Calculate total monthly expenses from budget data
        expenses = budget_data.get('expenses', {})
        monthly_expenses = sum(expenses.values()) if expenses else monthly_income * 0.8  # Default to 80% of income if no expenses
        
        # Calculate goal feasibility
        goal_analysis = calculate_goal_feasibility(
            target_amount=target_amount,
            current_savings=current_amount,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            target_date=target_date
        )
        
        # Store goal in session
        if 'goals' not in session:
            session['goals'] = []
            
        session['goals'].append({
            'name': goal_name,
            'target_amount': target_amount,
            'current_amount': current_amount,  # Store the current amount
            'time_frame': time_frame,
            'analysis': goal_analysis,
            'created_at': datetime.datetime.now().isoformat(),
            'target_date': target_date.isoformat()
        })
        
        return redirect(url_for('goals'))
    
    # For GET requests, show the goals page
    goals = session.get('goals', [])
    
    # Calculate progress for each goal
    for goal in goals:
        created_at = datetime.datetime.fromisoformat(goal['created_at'])
        target_date = created_at + datetime.timedelta(days=goal['time_frame']*30) if 'time_frame' in goal else None
        
        goal['progress'] = track_goal_progress(
            target_amount=goal['target_amount'],
            current_amount=goal.get('current_amount', 0),
            start_date=created_at,
            target_date=target_date
        )
    
    return render_template("goals.html", year=year, goals=goals)


@app.route("/elements")
def elements():
    year = get_current_year()
    return render_template("elements.html", year=year)


@app.route("/visualizations")
def visualizations():
    """Render the visualizations page."""
    year = get_current_year()
    return render_template("visualizations.html", year=year)


@app.route("/credit", methods=['GET', 'POST'])
def credit():
    year = get_current_year()
    
    if request.method == 'POST':
        credit_score = int(request.form.get('credit_score', 0))
        total_debt = float(request.form.get('total_debt', 0))
        credit_utilization = float(request.form.get('credit_utilization', 0))
        payment_history = request.form.get('payment_history', 'good')
        
        # Get income from session or use a default
        monthly_income = float(session.get('budget_data', {}).get('monthly_income', 5000))
        
        # Analyze credit and debt
        # Convert payment_history to a percentage (assuming 'good' = 100%, 'average' = 80%, etc.)
        on_time_payment_pct = 100 if payment_history == 'good' else 80
        
        # Set default values for required parameters not in the form
        avg_interest_rate = 15.0  # Default average interest rate
        open_credit_lines = 3      # Default number of open credit lines
        hard_inquiries_12m = 1     # Default number of hard inquiries
        
        credit_analysis = analyze_credit_and_debt(
            total_debt=total_debt,
            monthly_debt_payment=monthly_income * 0.1,  # 10% of monthly income as estimated payment
            avg_interest_rate=avg_interest_rate,
            credit_utilization_pct=credit_utilization,
            on_time_payment_pct=on_time_payment_pct,
            open_credit_lines=open_credit_lines,
            hard_inquiries_12m=hard_inquiries_12m
        )
        
        # Store in session
        session['credit_analysis'] = credit_analysis
        
        return render_template("credit.html", 
                             year=year,
                             credit_analysis=credit_analysis,
                             form_data=request.form)
    
    # For GET requests, show the form
    return render_template("credit.html", year=year)


@app.route("/download-csv")
def download_csv():
    # Create CSV data
    budget_data = session.get('budget_data', {})
    recommendations = session.get('recommendations', {})
    
    if not budget_data:
        return redirect(url_for('index'))
    
    # Create a string buffer to hold CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Category', 'Your Spending', 'Recommended', 'Difference', 'Status'])
    
    # Write expense data
    expenses = budget_data.get('expenses', {})
    for category, amount in expenses.items():
        recommended = recommendations.get('category_budgets', {}).get(category, 0)
        difference = amount - recommended
        status = 'Over' if difference > 0 else 'Under' if difference < 0 else 'On Target'
        writer.writerow([
            category.capitalize(),
            f'${amount:,.2f}',
            f'${recommended:,.2f}',
            f'${abs(difference):,.2f} {status}',
            status
        ])
    
    # Add summary
    writer.writerow([])
    writer.writerow(['Financial Health', 'Score', 'Status'])
    health = budget_data.get('financial_health', {})
    writer.writerow(['Overall Score', health.get('health_score', 0), health.get('status', '')])
    
    # Create response with CSV data
    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=financial_analysis.csv"}
    )


@app.route("/invest", methods=['GET', 'POST'])
def invest():
    year = get_current_year()
    
    if request.method == 'POST':
        # Get form data
        investment_amount = float(request.form.get('investment_amount', 0))
        risk_tolerance = request.form.get('risk_tolerance', 'moderate')
        investment_goal = request.form.get('investment_goal', 'retirement')
        time_horizon = int(request.form.get('time_horizon', 10))  # in years
        
        # Get user's financial data from session or use defaults
        monthly_income = float(session.get('budget_data', {}).get('monthly_income', 5000))
        monthly_savings = float(session.get('budget_data', {}).get('financial_health', {}).get('metrics', {}).get('monthly_savings', 1000))
        
        # Generate investment guidance
        investment_advice = generate_investment_guidance(
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
            investment_goal=investment_goal,
            time_horizon_years=time_horizon,
            monthly_income=monthly_income,
            monthly_savings=monthly_savings
        )
        
        return render_template("invest.html", 
                             year=year,
                             investment_advice=investment_advice,
                             form_data=request.form)
    
    # For GET requests, show the form
    return render_template("invest.html", year=year)


# API Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = make_prediction(data)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# This is required for Vercel
app = app  # This is required for Vercel to detect the Flask app

# This is the entry point for Vercel
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print("\n * Running on http://localhost:{} (Press Ctrl+C to quit)".format(port))
    app.run(host='127.0.0.1', port=port, debug=True)
