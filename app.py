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
    credit_inputs = None
    analysis = None
    
    if request.method == 'POST':
        try:
            # Get form data with defaults
            credit_inputs = {
                'total_debt': float(request.form.get('total_debt', 0)),
                'monthly_debt_payment': float(request.form.get('monthly_debt_payment', 0)),
                'avg_interest_rate': float(request.form.get('avg_interest_rate', 15.0)),
                'credit_utilization_pct': float(request.form.get('credit_utilization_pct', 30.0)),
                'on_time_payment_pct': float(request.form.get('on_time_payment_pct', 100)),
                'open_credit_lines': int(request.form.get('open_credit_lines', 3)),
                'hard_inquiries_12m': int(request.form.get('hard_inquiries_12m', 0))
            }
            
            # Call the credit advisor
            analysis = analyze_credit_and_debt(
                total_debt=credit_inputs['total_debt'],
                monthly_debt_payment=credit_inputs['monthly_debt_payment'],
                avg_interest_rate=credit_inputs['avg_interest_rate'],
                credit_utilization_pct=credit_inputs['credit_utilization_pct'],
                on_time_payment_pct=credit_inputs['on_time_payment_pct'],
                open_credit_lines=credit_inputs['open_credit_lines'],
                hard_inquiries_12m=credit_inputs['hard_inquiries_12m']
            )
            
            # Store in session for future reference
            session['credit_analysis'] = analysis
            
        except Exception as e:
            from flask import flash
            flash(f"Error processing your request: {str(e)}", "error")
    
    # For GET or if there was an error, show the form with any previous inputs
    return render_template(
        "credit.html", 
        year=year, 
        credit_inputs=credit_inputs or {},
        analysis=analysis
    )


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
    """Legacy investment route - redirects to the new advisor"""
    return redirect(url_for('investment_advisor'))

@app.route("/investment-advisor", methods=['GET', 'POST'])
def investment_advisor():
    """Enhanced investment advisor with modern UI and detailed recommendations"""
    year = get_current_year()
    
    # Get user data from session or use defaults
    user_data = {
        'monthly_income': float(session.get('budget_data', {}).get('monthly_income', 50000)),
        'age_range': request.form.get('age_range', '25-34'),
        'risk_tolerance': request.form.get('risk_tolerance', '').lower(),
        'time_horizon': int(request.form.get('time_horizon', 10)),
        'financial_health': session.get('budget_data', {}).get('financial_health', {})
    }
    
    # Generate investment guidance
    guidance = generate_investment_guidance(
        monthly_income=user_data['monthly_income'],
        age_range=user_data['age_range'],
        financial_health=user_data['financial_health'],
        risk_tolerance=user_data['risk_tolerance'] or None,
        time_horizon=user_data['time_horizon']
    )
    
    # Prepare data for the template
    template_data = {
        'year': year,
        'user_data': user_data,
        'guidance': guidance,
        'age_ranges': ['18-24', '25-34', '35-44', '45-54', '55+'],
        'risk_levels': [
            {'value': 'conservative', 'label': 'Conservative'},
            {'value': 'moderate', 'label': 'Moderate'},
            {'value': 'aggressive', 'label': 'Aggressive'}
        ],
        'time_horizons': [
            {'years': 1, 'label': '1 year'},
            {'years': 3, 'label': '3 years'},
            {'years': 5, 'label': '5 years'},
            {'years': 10, 'label': '10 years'},
            {'years': 15, 'label': '15 years'},
            {'years': 20, 'label': '20+ years'}
        ]
    }
    
    return render_template("investment_advisor_new.html", **template_data)
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
