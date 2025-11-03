# OakLedger

<div align="center">

<img src="https://img.shields.io/badge/OakLedger-Personal%20Finance%20Assistant-0f172a?style=for-the-badge&logo=python&logoColor=3776AB" alt="OakLedger Banner">

### Smart Personal Finance on the Web — Built with Flask + Jinja + Chart.js  
#### Budgeting • Financial Health • Goals • Credit/Debt • Investment Guidance

---

OakLedger is a lightweight Flask web app that helps you:

- Analyze your financial health from income and expenses.
- Get India‑context budget recommendations (age, family, location sensitivity).
- Plan goals and track progress automatically.
- Review credit/debt metrics and get actionable advice.
- Explore low‑risk vs growth Investment Guidance.


[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white&style=flat-square)](https://flask.palletsprojects.com/)
[![Jinja](https://img.shields.io/badge/Jinja2-Templating-B41717?logo=jinja&logoColor=white&style=flat-square)](https://palletsprojects.com/p/jinja/)
[![Chart.js](https://img.shields.io/badge/Chart.js-Visualizations-FF6384?logo=chartdotjs&logoColor=white&style=flat-square)](https://www.chartjs.org/)

</div>

---

## Features

- **Budget Recommendations (India‑context)**
  - Location cost‑of‑living tiers, age range, family size, income tier.
  - Percent allocations with currency amounts and a pie chart.

- **Financial Health**
  - Health score and status with metrics (savings rate, expense ratio).
  - Targeted recommendations by category.

- **Goals Planner**
  - Add goals with target amount/date; feasibility and progress tracking.

- **Credit & Debt Advisor**
  - Inputs like debt, payment, interest, utilization, inquiries.
  - Suggestions to improve credit profile and debt management.

- **Investment Guidance (New)**
  - Simple split into Low‑Risk vs Growth buckets.
  - Age/savings‑rate‑aware risk profile. Short‑term goal nudges.
  - Concrete instruments (e.g., Liquid/Overnight, FD, PPF/EPF, Nifty Index, ELSS, NPS).

- **Sample Profiles**
  - Pre‑filled data to preview working of the website and taking reference.

---

## Tech Stack

- Backend: Flask + Python
- Templating: Jinja2
- UI: HTML5UP Phantom theme (custom darkened styles)
- Charts: Chart.js
- Storage: Flask session (no DB)
- Styling: Custom CSS, Font Awesome via CDN

---

## Project Structure

```text
c:\OakLedger
 ┣ static/
 ┃ ┣ css/
 ┃ ┃ ┣ main.css
 ┃ ┃ ┗ noscript.css
 ┃ ┣ js/
 ┃ ┗ assets/
 ┣ templates/
 ┃ ┣ header.html
 ┃ ┣ footer.html
 ┃ ┣ index.html
 ┃ ┣ generic.html
 ┃ ┣ goals.html
 ┃ ┣ credit.html
 ┃ ┗ investment.html
 ┣ budget_calc.py
 ┣ financial_analyzer.py
 ┣ credit_advisor.py
 ┣ goal_planner.py
 ┣ investment_advisor.py
 ┣ sample_profiles.py
 ┣ main.py
 ┗ README.md
