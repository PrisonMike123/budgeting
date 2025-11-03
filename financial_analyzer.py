def calculate_financial_health(monthly_income, expenses):
    """
    Calculate financial health score and provide recommendations based on income and expenses.
    
    Parameters:
    monthly_income (float): Monthly income
    expenses (dict): Dictionary of expenses by category
    
    Returns:
    dict: Financial health analysis including:
        - health_score (0-100)
        - status (Critical, Poor, Fair, Good, Excellent)
        - metrics (dict of financial metrics)
        - recommendations (list of suggestions)
    """
    # Calculate total expenses
    total_expenses = sum(expenses.values())
    
    # Calculate key financial metrics
    savings_rate = (monthly_income - total_expenses) / monthly_income * 100
    expense_to_income_ratio = (total_expenses / monthly_income) * 100
    
    # Initialize metrics dictionary
    metrics = {
        "savings_rate": round(savings_rate, 2),
        "expense_to_income_ratio": round(expense_to_income_ratio, 2),
        "monthly_savings": round(monthly_income - total_expenses, 2)
    }
    
    # Calculate health score (0-100)
    health_score = calculate_health_score(savings_rate, expense_to_income_ratio)
    
    # Determine status based on health score
    status = get_health_status(health_score)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics, expenses)
    
    return {
        "health_score": health_score,
        "status": status,
        "metrics": metrics,
        "recommendations": recommendations
    }

def calculate_health_score(savings_rate, expense_ratio):
    """Calculate a health score from 0-100 based on financial metrics."""
    # Base score on savings rate (weight: 60%) and expense ratio (weight: 40%)
    savings_score = min(100, max(0, savings_rate * 2))  # 15% savings = 30 points
    expense_score = min(100, max(0, (100 - expense_ratio) * 1.5))  # 70% expenses = 45 points
    
    return round(savings_score * 0.6 + expense_score * 0.4)

def get_health_status(score):
    """Convert numerical score to status category."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    elif score >= 40:
        return "Poor"
    else:
        return "Critical"

def generate_recommendations(metrics, expenses):
    """Generate personalized recommendations based on financial metrics."""
    recommendations = []
    
    # Check savings rate
    if metrics["savings_rate"] < 20:
        recommendations.append({
            "category": "Savings",
            "suggestion": "Aim to save at least 20% of your monthly income",
            "priority": "High"
        })
    
    # Check expense ratio
    if metrics["expense_to_income_ratio"] > 80:
        recommendations.append({
            "category": "Expenses",
            "suggestion": "Your expenses are high relative to income. Review non-essential spending.",
            "priority": "High"
        })
    
    # Check individual expense categories
    for category, amount in expenses.items():
        if category == "housing" and amount > metrics["monthly_savings"] * 0.3:
            recommendations.append({
                "category": "Housing",
                "suggestion": "Housing costs exceed 30% of income. Consider more affordable options.",
                "priority": "Medium"
            })
        elif category == "discretionary" and amount > metrics["monthly_savings"] * 0.2:
            recommendations.append({
                "category": "Discretionary",
                "suggestion": "Consider reducing discretionary spending to increase savings.",
                "priority": "Medium"
            })
    
    return recommendations