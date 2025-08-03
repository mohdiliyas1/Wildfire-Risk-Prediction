// Scroll animation for cards
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add('show');
  });
});

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

// Bar chart – Feature Importance (Example values)
const barCtx = document.getElementById('barChart');
new Chart(barCtx, {
  type: 'bar',
  data: {
    labels: ['Temperature', 'Humidity', 'Wind Speed', 'Rainfall'],
    datasets: [{
      label: 'Feature Importance Score',
      data: [0.35, 0.25, 0.20, 0.15],
      backgroundColor: ['#ff6f61', '#ffa07a', '#87ceeb', '#90ee90'],
      borderRadius: 10
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Importance', color: '#555' }
      }
    }
  }
});

// Pie chart – Risk Distribution (Example)
const pieCtx = document.getElementById('pieChart');
new Chart(pieCtx, {
  type: 'pie',
  data: {
    labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
    datasets: [{
      data: [40, 35, 25],
      backgroundColor: ['#90ee90', '#ffd700', '#ff6f61'],
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'bottom' }
    }
  }
});
