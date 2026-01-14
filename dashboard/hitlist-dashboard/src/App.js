import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [buyOpportunities, setBuyOpportunities] = useState([]);
  const [sellOpportunities, setSellOpportunities] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/api/hitlist')
      .then(response => {
        setBuyOpportunities(response.data.buy_opportunities || []);
        setSellOpportunities(response.data.sell_opportunities || []);
      })
      .catch(error => {
        setError('Error fetching hitlist data. Please ensure the backend is running.');
        console.error('There was an error fetching the data!', error);
      });
  }, []);

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">PSX Trading Hitlist</h1>

      {error && <div className="alert alert-danger">Failed to edit, 0 occurrences found for old_string (import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit src/App.js and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
). Original old_string was (import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit src/App.js and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
) in /Users/macair2020/Desktop/Algo_Trading/dashboard/hitlist-dashboard/src/App.js. No edits made. The exact text in old_string was not found. Ensure you're not escaping content incorrectly and check whitespace, indentation, and context. Use read_file tool to verify.</div>}

      <div className="card">
        <div className="card-header">
          <h2>Top Buy Opportunities</h2>
        </div>
        <div className="card-body">
          <table className="table table-striped">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Entry</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {buyOpportunities.length > 0 ? (
                buyOpportunities.map((stock, index) => (
                  <tr key={index}>
                    <td>{stock.symbol}</td>
                    <td><span className="badge bg-success">{stock.signal}</span></td>
                    <td>{stock.confidence.toFixed(2)}%</td>
                    <td>{stock.entry.toFixed(2)}</td>
                    <td>{stock.stop_loss.toFixed(2)}</td>
                    <td>{stock.take_profit.toFixed(2)}</td>
                    <td>{stock.reason}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" className="text-center">No buy opportunities found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mt-5">
        <div className="card-header">
          <h2>Top Sell Opportunities</h2>
        </div>
        <div className="card-body">
          <table className="table table-striped">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Entry</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {sellOpportunities.length > 0 ? (
                sellOpportunities.map((stock, index) => (
                  <tr key={index}>
                    <td>{stock.symbol}</td>
                    <td><span className="badge bg-danger">{stock.signal}</span></td>
                    <td>{stock.confidence.toFixed(2)}%</td>
                    <td>{stock.entry.toFixed(2)}</td>
                    <td>{stock.stop_loss.toFixed(2)}</td>
                    <td>{stock.take_profit.toFixed(2)}</td>
                    <td>{stock.reason}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" className="text-center">No sell opportunities found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App;
