import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './index.css';

const MARKET_IMAGES = [
  'https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1518186285589-2f7649de83e0?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1000&auto=format&fit=crop'  
];

const ECO_IMAGES = [
  'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1502082553048-f009c37129b9?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=1000&auto=format&fit=crop', 
  'https://images.unsplash.com/photo-1511497584788-876760111969?q=80&w=1000&auto=format&fit=crop'  
];

function App() {
  const [formData, setFormData] = useState({ 
    lat: 28.61, 
    lon: 77.20, 
    area: 10, 
    eff: 0.18, 
    days: 365 
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  
  const calculatorRef = useRef(null);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentImageIndex((prevIndex) => (prevIndex + 1) % MARKET_IMAGES.length);
    }, 5000); 
    return () => clearInterval(intervalId);
  }, []);

  const scrollToCalculator = () => {
    calculatorRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const calculate = async () => {
    if (loading) return;
    setLoading(true);

    try {
      const response = await axios.post('/api/calculate', formData);
      setResult(response.data);
      setLoading(false);
    } catch (err) {
      console.log("Backend not found, using demo mode");
      const timeframeFactor = formData.days / 365;
      const areaFactor = formData.area / 10;
      
      setTimeout(() => {
        setResult({
          carbon_credits: (12.5 * timeframeFactor * areaFactor).toFixed(2),
          co2_avoided_kg: (12500 * timeframeFactor * areaFactor).toFixed(0),
          total_yield_kwh: (15400 * timeframeFactor * areaFactor).toFixed(0),
          z_factor: 0.82
        });
        setLoading(false);
      }, 1500); 
    }
  };

  return (
    <div>
      {/* --- Section 1: Hero (NO BOX, DIRECT TEXT) --- */ }
      <section className="hero-section">
        <div className="hero-content">
          <h1>Monetize Your <br />Solar Impact</h1>
          <p>
            Transform your solar energy generation into certified, tradeable carbon credits. 
            Use our institutional-grade AI to forecast your potential revenue instantly.
          </p>
          <button className="scroll-btn" onClick={scrollToCalculator}>
            Start Calculation ↓
          </button>
        </div>
      </section>

      {/* --- Section 2: Calculator (DARK) --- */ }
      <section className="calculator-section" ref={calculatorRef}>
        
        <div 
          className="side-visual"
          style={{ backgroundImage: `url(${MARKET_IMAGES[currentImageIndex]})` }}
        >
           <div className="visual-label">Market Analytics</div>
        </div>

        <div className="dashboard-container">
          <div className="glass-card">
            <header>
              <h2>SolarCarbon AI</h2>
              <p>Institutional Grade Prediction</p>
            </header>

            <div className="input-section">
              <div className="input-grid">
                <div className="field">
                  <label>1. Latitude</label>
                  <input type="number" step="0.01" value={formData.lat} onChange={(e) => setFormData({...formData, lat: parseFloat(e.target.value)})} />
                </div>
                <div className="field">
                  <label>2. Longitude</label>
                  <input type="number" step="0.01" value={formData.lon} onChange={(e) => setFormData({...formData, lon: parseFloat(e.target.value)})} />
                </div>
                <div className="field full-width">
                  <label>3. Panel Area (m²)</label>
                  <input type="number" value={formData.area} onChange={(e) => setFormData({...formData, area: parseFloat(e.target.value)})} />
                </div>
                <div className="field full-width">
                  <label>4. Timeframe</label>
                  <div className="select-wrapper">
                    <select value={formData.days} onChange={(e) => setFormData({...formData, days: parseInt(e.target.value)})}>
                      <optgroup label="Short Term (Months)">
                        {Array.from({ length: 11 }, (_, i) => i + 1).map((month) => (
                          <option key={`m-${month}`} value={month * 30}>{month} Month{month > 1 ? 's' : ''}</option>
                        ))}
                      </optgroup>
                      <optgroup label="Long Term (Years)">
                        {Array.from({ length: 25 }, (_, i) => i + 1).map((year) => (
                          <option key={`y-${year}`} value={year * 365}>{year} Year{year > 1 ? 's' : ''}</option>
                        ))}
                      </optgroup>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <button className="calc-btn" onClick={calculate} disabled={loading} style={{ opacity: loading ? 0.7 : 1, cursor: loading ? 'wait' : 'pointer' }}>
              {loading ? "Analyzing..." : "Generate Credit Report"}
            </button>

            {result && (
              <div className="results-panel">
                <div className="credit-hero">
                  <span className="result-label">Estimated Assets</span>
                  <h3>{result.carbon_credits} Credits</h3>
                </div>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Avoided Emissions</div>
                    <div className="stat-val">{result.co2_avoided_kg} kg</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Energy Yield</div>
                    <div className="stat-val">{result.total_yield_kwh} kWh</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div 
          className="side-visual"
          style={{ backgroundImage: `url(${ECO_IMAGES[currentImageIndex]})` }}
        >
          <div className="visual-label">Ecological Impact</div>
        </div>

      </section>
    </div>
  );
}

export default App;