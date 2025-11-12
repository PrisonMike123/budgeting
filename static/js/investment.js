// Initialize charts when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded');
        return;
    }

    // Initialize allocation chart
    const allocCtx = document.getElementById('allocationChart');
    if (allocCtx) {
        const chartData = JSON.parse(allocCtx.dataset.chartData);
        new Chart(allocCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low-Risk', 'Growth'],
                datasets: [{
                    data: [chartData.lowRisk, chartData.growth],
                    backgroundColor: ['#4CAF50', '#2196F3'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            padding: 15
                        }
                    }
                }
            }
        });
    }

    // Initialize projection chart
    const projCtx = document.getElementById('projectionChart');
    if (projCtx) {
        const chartData = JSON.parse(projCtx.dataset.chartData);
        const monthlyInvestment = (chartData.monthlyIncome * chartData.totalAllocationPct) / 100;
        
        function calculateProjection(years, annualReturn) {
            let total = 0;
            return years.map(year => {
                total = (total + (monthlyInvestment * 12)) * (1 + annualReturn);
                return Math.round(total);
            });
        }

        const years = [0, 1, 3, 5, 10, 15, 20];
        
        new Chart(projCtx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'Conservative (3% return)',
                        data: calculateProjection(years, 0.03),
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Moderate (6% return)',
                        data: calculateProjection(years, 0.06),
                        borderColor: '#FFC107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Aggressive (9% return)',
                        data: calculateProjection(years, 0.09),
                        borderColor: '#F44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', {
                                        style: 'currency',
                                        currency: 'USD',
                                        maximumFractionDigits: 0
                                    }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }
});
