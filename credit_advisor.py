import math


def _months_to_payoff(principal, monthly_payment, annual_rate_percent):
    r = (annual_rate_percent or 0) / 100.0 / 12.0
    if principal <= 0:
        return 0
    if monthly_payment <= 0:
        return None
    if r <= 0:
        return math.ceil(principal / monthly_payment)
    if monthly_payment <= principal * r:
        return None
    n = math.log(monthly_payment / (monthly_payment - principal * r)) / math.log(1 + r)
    return math.ceil(n)


def analyze_credit_and_debt(
    total_debt,
    monthly_debt_payment,
    avg_interest_rate,
    credit_utilization_pct,
    on_time_payment_pct,
    open_credit_lines,
    hard_inquiries_12m,
):
    months = _months_to_payoff(total_debt, monthly_debt_payment, avg_interest_rate)

    strategy = "avalanche" if avg_interest_rate >= 10 else "snowball"
    strategy_reason = (
        "Focus higher-APR balances first to minimize interest" if strategy == "avalanche" else "Build momentum by closing small balances first"
    )

    utilization_adv = []
    if credit_utilization_pct > 50:
        utilization_adv.append("Reduce utilization below 50% to stop score drag")
    if credit_utilization_pct > 30:
        utilization_adv.append("Aim for <30% utilization; <10% is ideal")
    if not utilization_adv:
        utilization_adv.append("Keep utilization low and consistent (<10% ideal)")

    payment_hist_adv = []
    if on_time_payment_pct < 95:
        payment_hist_adv.append("Automate payments to ensure 100% on-time history")
    else:
        payment_hist_adv.append("Maintain perfect on-time payments; this is the top factor")

    mix_adv = []
    if open_credit_lines < 2:
        mix_adv.append("Consider a secured or low-fee card to build depth before new loans")
    else:
        mix_adv.append("Avoid unnecessary new accounts; age and stability help")

    inquiry_adv = []
    if hard_inquiries_12m >= 3:
        inquiry_adv.append("Pause new applications for 6–12 months to let inquiries age")
    else:
        inquiry_adv.append("Batch rate-shopping within 14–45 days if needed")

    general = []
    if months is None:
        general.append("Increase monthly payment; current amount may not cover interest")
    elif months > 60:
        general.append("Consider refinancing or consolidating to lower APR")
    elif months > 24:
        general.append("Add small extra payments to reduce term and total interest")

    result = {
        "payoff_strategy": strategy,
        "strategy_reason": strategy_reason,
        "estimated_months_to_payoff": months,
        "utilization_advice": utilization_adv,
        "payment_history_advice": payment_hist_adv,
        "credit_mix_advice": mix_adv,
        "inquiries_advice": inquiry_adv,
        "general_recommendations": general,
    }
    return result
