document.getElementById('fireForm').onsubmit = async (e) => {
  e.preventDefault();
  const data = {
    X: parseFloat(document.getElementById('X').value),
    Y: parseFloat(document.getElementById('Y').value),
    FFMC: parseFloat(document.getElementById('FFMC').value),
    DMC: parseFloat(document.getElementById('DMC').value),
    DC: parseFloat(document.getElementById('DC').value),
    ISI: parseFloat(document.getElementById('ISI').value),
    temp: parseFloat(document.getElementById('temp').value),
    RH: parseFloat(document.getElementById('RH').value),
    wind: parseFloat(document.getElementById('wind').value),
    rain: parseFloat(document.getElementById('rain').value)
  };

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await res.json();
  document.getElementById('result').innerText = result.fire_risk === 1 
    ? "🔥 High Fire Risk Detected" 
    : "✅ No Fire Risk";
};
