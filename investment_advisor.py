import datetime


def generate_investment_guidance(monthly_income, age_range, financial_health=None, goals=None, risk_tolerance=None, time_horizon=10):
    """
    Provide comprehensive investment guidance with detailed allocations.

    Args:
        monthly_income (float): Monthly income in ₹
        age_range (str): Age range bracket (e.g., '25-34')
        financial_health (dict, optional): Dict containing financial metrics
        goals (list, optional): List of financial goals
        risk_tolerance (str, optional): User's risk tolerance ('conservative', 'moderate', 'aggressive')
        time_horizon (int, optional): Investment time horizon in years (default: 10)

    Returns:
        dict: Comprehensive investment guidance including risk profile, allocations, and recommendations
    """
    # Initialize notes list
    notes = []
    # Default risk profile based on age if not provided
    if not risk_tolerance:
        age_to_profile = {
            '18-24': 'aggressive',
            '25-34': 'aggressive',
            '35-44': 'moderate',
            '45-54': 'moderate',
            '55+': 'conservative',
        }
        risk_profile = age_to_profile.get(age_range, 'moderate')
    else:
        risk_profile = risk_tolerance.lower()

    # Adjust risk profile based on time horizon
    if time_horizon < 3:
        risk_profile = 'conservative'
    elif time_horizon < 7:
        if risk_profile == 'aggressive':
            risk_profile = 'moderate'
    
    # Adjust based on financial health if available
    if financial_health and isinstance(financial_health, dict):
        metrics = financial_health.get('metrics', {})
        savings_rate = metrics.get('savings_rate')
        
        if isinstance(savings_rate, (int, float)):
            if savings_rate < 10 and risk_profile == 'aggressive':
                risk_profile = 'moderate'
            elif savings_rate > 25 and risk_profile == 'moderate':
                risk_profile = 'aggressive'

    # Determine allocation based on risk profile
    allocation = {
        'conservative': {'low_risk': 70, 'growth': 30},
        'moderate': {'low_risk': 50, 'growth': 50},
        'aggressive': {'low_risk': 30, 'growth': 70}
    }[risk_profile]

    # Investment options with enhanced details
    investment_options = {
        'low_risk': [
            {
                'name': 'Emergency Fund',
                'description': 'Liquid assets for 3-6 months of expenses',
                'type': 'Liquid/Overnight Fund',
                'risk': 'Very Low',
                'returns': '3-5%',
                'liquidity': 'High',
                'tax_efficiency': 'Medium',
                'weight': 0.35,
                'notes': 'Keep in high-yield savings or liquid funds for easy access.'
            },
            {
                'name': 'Fixed Income',
                'description': 'Bank FDs, Corporate Bonds, and Debt Funds',
                'type': 'Fixed Income',
                'risk': 'Low',
                'returns': '5-7%',
                'liquidity': 'Medium',
                'tax_efficiency': 'Low',
                'weight': 0.35,
                'notes': 'Consider tax-efficient options like debt mutual funds for better post-tax returns.'
            },
            {
                'name': 'Hybrid Funds',
                'description': 'Balanced Advantage or Conservative Hybrid Funds',
                'type': 'Hybrid',
                'risk': 'Low to Medium',
                'returns': '7-9%',
                'liquidity': 'Medium',
                'tax_efficiency': 'High',
                'weight': 0.20,
                'notes': 'Good for conservative investors seeking slightly higher returns than pure debt.'
            },
            {
                'name': 'Gold',
                'description': 'Sovereign Gold Bonds or Gold ETFs',
                'type': 'Commodity',
                'risk': 'Medium',
                'returns': '8-10%',
                'liquidity': 'High',
                'tax_efficiency': 'High',
                'weight': 0.10,
                'notes': 'Good hedge against inflation and market volatility.'
            }
        ],
        'growth': [
            {
                'name': 'Equity Index Funds',
                'description': 'Nifty 50, Nifty Next 50 Index Funds',
                'type': 'Equity',
                'risk': 'Medium',
                'returns': '10-12%',
                'liquidity': 'High',
                'tax_efficiency': 'High',
                'weight': 0.40,
                'notes': 'Low-cost way to get broad market exposure.'
            },
            {
                'name': 'Sectoral/Thematic Funds',
                'description': 'Technology, Healthcare, Banking, etc.',
                'type': 'Equity',
                'risk': 'High',
                'returns': '12-15%',
                'liquidity': 'High',
                'tax_efficiency': 'High',
                'weight': 0.20,
                'notes': 'Higher risk, higher return potential. Limit exposure.'
            },
            {
                'name': 'Small & Mid Cap Funds',
                'description': 'Focused on smaller companies with growth potential',
                'type': 'Equity',
                'risk': 'High',
                'returns': '12-18%',
                'liquidity': 'Medium',
                'tax_efficiency': 'High',
                'weight': 0.25,
                'notes': 'Higher volatility but potential for superior long-term returns.'
            },
            {
                'name': 'International Equity',
                'description': 'US/Global index funds or ETFs',
                'type': 'International',
                'risk': 'Medium to High',
                'returns': '10-14%',
                'liquidity': 'Medium',
                'tax_efficiency': 'Medium',
                'weight': 0.15,
                'notes': 'Provides geographic diversification and currency hedge.'
            }
        ]
    }

    # Calculate amounts for each category
    total_investment = monthly_income * (allocation['low_risk'] + allocation['growth']) / 100
    
    # Allocate within categories
    def allocate_within_category(category, total_amount):
        category_total = total_amount * (allocation[category] / 100)
        options = investment_options[category]
        total_weight = sum(opt['weight'] for opt in options)
        
        allocations = []
        for opt in options:
            amount = (opt['weight'] / total_weight) * category_total
            allocations.append({
                'name': opt['name'],
                'description': opt['description'],
                'type': opt['type'],
                'risk': opt['risk'],
                'expected_returns': opt['returns'],
                'liquidity': opt['liquidity'],
                'tax_efficiency': opt['tax_efficiency'],
                'percentage': round(opt['weight'] * 100, 1),
                'amount': round(amount, 2),
                'notes': opt['notes']
            })
        return allocations

    # Generate risk description
    risk_descriptions = {
        'conservative': {
            'title': 'Conservative',
            'description': 'You prefer to protect your capital and are willing to accept lower returns for greater stability. Your portfolio is focused on capital preservation with minimal exposure to market volatility.',
            'suitability': 'Best for short-term goals (1-3 years) or those with low risk tolerance.'
        },
        'moderate': {
            'title': 'Moderate',
            'description': 'You have a balanced approach to risk and return. Your portfolio includes a mix of growth and income-generating assets, providing a good balance between stability and growth potential.',
            'suitability': 'Ideal for medium-term goals (3-7 years) or those with moderate risk tolerance.'
        },
        'aggressive': {
            'title': 'Aggressive',
            'description': 'You have a high tolerance for risk and are comfortable with market volatility. Your portfolio is growth-oriented with a focus on equities and other high-growth assets to maximize long-term returns.',
            'suitability': 'Best for long-term goals (7+ years) or those with high risk tolerance.'
        }
    }

    # Generate projections
    def calculate_projections(principal, monthly_contribution, years, return_rate):
        monthly_rate = (1 + return_rate) ** (1/12) - 1
        months = years * 12
        future_value = principal * (1 + monthly_rate) ** months
        future_value += monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        return round(future_value, 2)

    # Calculate projections for different scenarios
    projection_years = [1, 3, 5, 10, 15, 20]
    projection_scenarios = [
        {'label': 'Conservative', 'rate': 0.06},
        {'label': 'Moderate', 'rate': 0.09},
        {'label': 'Aggressive', 'rate': 0.12}
    ]
    
    projections = {}
    for scenario in projection_scenarios:
        projections[scenario['label']] = [
            calculate_projections(
                0,  # Starting principal
                total_investment,  # Monthly investment
                year,
                scenario['rate']
            )
            for year in projection_years
        ]

    # Prepare the final response
    # Check for short-term goals if goals are provided
    if goals and isinstance(goals, list):
        try:
            soon = False
            for g in goals:
                target_date = g.get('target_date')
                # Templates may serialize dates to string; handle both
                if isinstance(target_date, str):
                    try:
                        target_date = datetime.datetime.fromisoformat(target_date)
                    except Exception:
                        target_date = None
                
                if target_date:
                    months = (target_date.year - datetime.datetime.now().year) * 12 + (target_date.month - datetime.datetime.now().month)
                    if months <= 24:
                        soon = True
                        break
            
            if soon and risk_profile != 'conservative':
                notes.append('Short-term goals detected (≤ 24 months). Consider shifting 10% more towards low-risk instruments until goals are funded.')
        except Exception as e:
            print(f"Error processing goals: {e}")
            pass

    # General best-practice notes for India context
    notes.extend([
        'Prioritize building a 3–6 month emergency fund before increasing equity exposure.',
        'Rebalance portfolio annually to maintain your chosen risk split.',
        'Use low-cost index funds as core holdings; add actives only if they fit your plan.',
    ])

    # Prepare the response dictionary with all required fields
    return {
        'risk_profile': {
            'level': risk_profile,
            'title': risk_descriptions[risk_profile]['title'],
            'description': risk_descriptions[risk_profile]['description'],
            'suitability': risk_descriptions[risk_profile]['suitability']
        },
        'allocation': {
            'low_risk': allocation['low_risk'],
            'growth': allocation['growth'],
            'total_investment': round(total_investment, 2),
            'monthly_investment': round(total_investment, 2)
        },
        'split': {
            'low_risk_pct': allocation['low_risk'],
            'growth_pct': allocation['growth']
        },
        'recommendations': {
            'low_risk': allocate_within_category('low_risk', total_investment),
            'growth': allocate_within_category('growth', total_investment)
        },
        'projections': {
            'years': projection_years,
            'scenarios': projections
        },
        'time_horizon': time_horizon,
        'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notes': notes
    }
