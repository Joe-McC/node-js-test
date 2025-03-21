import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import './styles/node-styles.css';

const ParameterTuningNode = ({ data, isConnectable }) => {
  const [tuningMethod, setTuningMethod] = useState(data.tuningMethod || 'grid');
  const [parameters, setParameters] = useState(data.parameters || [
    { name: 'learning_rate', min: 0.0001, max: 0.1, step: 0.001, value: 0.001 },
    { name: 'batch_size', min: 8, max: 256, step: 8, value: 32 },
    { name: 'dropout', min: 0, max: 0.5, step: 0.05, value: 0.2 }
  ]);
  const [tuningStatus, setTuningStatus] = useState(data.tuningStatus || 'Not Started');
  const [tuningLogs, setTuningLogs] = useState(data.tuningLogs || []);
  const [iterations, setIterations] = useState(data.iterations || 10);
  const [isRunning, setIsRunning] = useState(false);
  const [tuningProgress, setTuningProgress] = useState(data.tuningProgress || 0);
  const [bestParameters, setBestParameters] = useState(data.bestParameters || null);
  const [bestScore, setBestScore] = useState(data.bestScore || null);
  const [tuningHistory, setTuningHistory] = useState(data.tuningHistory || []);
  const [objective, setObjective] = useState(data.objective || 'accuracy');

  const tuningMethods = [
    { value: 'grid', label: 'Grid Search' },
    { value: 'random', label: 'Random Search' },
    { value: 'bayesian', label: 'Bayesian Optimization' },
    { value: 'genetic', label: 'Genetic Algorithm' }
  ];
  
  const objectives = [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'f1', label: 'F1 Score' },
    { value: 'auc', label: 'AUC' },
    { value: 'recall', label: 'Recall' },
    { value: 'precision', label: 'Precision' },
    { value: 'log_loss', label: 'Log Loss' },
    { value: 'rmse', label: 'RMSE' }
  ];

  useEffect(() => {
    // Update node data when state changes
    data.updateNodeData(data.id, {
      tuningMethod,
      parameters,
      tuningStatus,
      tuningLogs,
      iterations,
      tuningProgress,
      bestParameters,
      bestScore,
      tuningHistory,
      objective,
      isConfigured: Boolean(tuningMethod)
    });
  }, [
    tuningMethod,
    parameters,
    tuningStatus,
    tuningLogs,
    iterations,
    tuningProgress,
    bestParameters,
    bestScore,
    tuningHistory,
    objective
  ]);

  const updateParameter = (index, field, value) => {
    const updatedParams = [...parameters];
    updatedParams[index] = { ...updatedParams[index], [field]: value };
    setParameters(updatedParams);
  };

  const addParameter = () => {
    setParameters([
      ...parameters,
      { name: `param_${parameters.length + 1}`, min: 0, max: 1, step: 0.1, value: 0.5 }
    ]);
  };

  const removeParameter = (index) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const startTuning = () => {
    if (parameters.length === 0) {
      alert('Please add at least one parameter to tune');
      return;
    }
    
    setIsRunning(true);
    setTuningStatus('Running');
    setTuningProgress(0);
    setTuningLogs([...tuningLogs, `[${new Date().toLocaleTimeString()}] Starting parameter tuning using ${tuningMethod}`]);
    
    // Clear previous results
    setTuningHistory([]);
    setBestParameters(null);
    setBestScore(null);
    
    // Simulate parameter tuning
    let progress = 0;
    const interval = setInterval(() => {
      progress += 100 / iterations;
      
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        
        // Generate best parameters
        const best = parameters.reduce((obj, param) => {
          obj[param.name] = Number(
            (param.min + Math.random() * (param.max - param.min)).toFixed(4)
          );
          return obj;
        }, {});
        
        setBestParameters(best);
        setBestScore(Number((0.8 + Math.random() * 0.15).toFixed(4)));
        setIsRunning(false);
        setTuningStatus('Completed');
        
        setTuningLogs(prev => [
          ...prev,
          `[${new Date().toLocaleTimeString()}] Parameter tuning completed`,
          `[${new Date().toLocaleTimeString()}] Best ${objective}: ${bestScore}`,
          `[${new Date().toLocaleTimeString()}] Best parameters found`
        ]);
      } else {
        // Add a history point occasionally
        if (Math.random() > 0.7 || progress % 20 < 5) {
          const score = Number((0.5 + Math.random() * 0.4).toFixed(4));
          const params = parameters.reduce((obj, param) => {
            obj[param.name] = Number(
              (param.min + Math.random() * (param.max - param.min)).toFixed(4)
            );
            return obj;
          }, {});
          
          setTuningHistory(prev => [...prev, { iteration: prev.length + 1, params, score }]);
          
          setTuningLogs(prev => [
            ...prev,
            `[${new Date().toLocaleTimeString()}] Iteration ${tuningHistory.length + 1}, ${objective}: ${score}`
          ]);
        }
      }
      
      setTuningProgress(progress);
    }, 500);
  };

  return (
    <div className="ml-node parameter-tuning-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="node-content">
        <h4>{data.label || 'Parameter Tuning'}</h4>
        
        <div className="input-group">
          <label>Tuning Method:</label>
          <select 
            value={tuningMethod} 
            onChange={(e) => setTuningMethod(e.target.value)}
            className="nodrag"
          >
            <option value="">Select Method</option>
            {tuningMethods.map(method => (
              <option key={method.value} value={method.value}>{method.label}</option>
            ))}
          </select>
        </div>
        
        <div className="input-group">
          <label>Optimization Objective:</label>
          <select 
            value={objective} 
            onChange={(e) => setObjective(e.target.value)}
            className="nodrag"
          >
            {objectives.map(obj => (
              <option key={obj.value} value={obj.value}>{obj.label}</option>
            ))}
          </select>
        </div>
        
        <div className="input-group">
          <label>Iterations:</label>
          <input 
            type="number" 
            min="1" 
            max="100"
            value={iterations}
            onChange={(e) => setIterations(parseInt(e.target.value))}
            className="nodrag"
          />
        </div>
        
        <div className="parameters-list">
          <h5>Parameters to Tune</h5>
          
          {parameters.map((param, index) => (
            <div key={index} className="parameter-item">
              <div className="param-row">
                <input
                  type="text"
                  value={param.name}
                  onChange={(e) => updateParameter(index, 'name', e.target.value)}
                  placeholder="Parameter name"
                  className="param-name nodrag"
                />
                <button 
                  onClick={() => removeParameter(index)}
                  className="small-button remove-btn">
                  ✕
                </button>
              </div>
              
              <div className="param-row">
                <div className="param-range">
                  <label>Min:</label>
                  <input
                    type="number"
                    value={param.min}
                    onChange={(e) => updateParameter(index, 'min', parseFloat(e.target.value))}
                    className="nodrag"
                  />
                  
                  <label>Max:</label>
                  <input
                    type="number"
                    value={param.max}
                    onChange={(e) => updateParameter(index, 'max', parseFloat(e.target.value))}
                    className="nodrag"
                  />
                  
                  <label>Step:</label>
                  <input
                    type="number"
                    value={param.step}
                    onChange={(e) => updateParameter(index, 'step', parseFloat(e.target.value))}
                    className="nodrag"
                  />
                </div>
              </div>
            </div>
          ))}
          
          <button onClick={addParameter} className="add-param-btn">
            + Add Parameter
          </button>
        </div>
        
        <div className="tuning-controls">
          <button 
            onClick={startTuning} 
            disabled={isRunning || !tuningMethod || parameters.length === 0}
            className="tune-button">
            {isRunning ? 'Tuning...' : 'Start Tuning'}
          </button>
          
          {tuningProgress > 0 && (
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{width: `${tuningProgress}%`}}>
              </div>
              <span>{tuningProgress.toFixed(0)}%</span>
            </div>
          )}
          
          <div className="status">Status: {tuningStatus}</div>
        </div>
        
        {bestParameters && (
          <div className="best-params">
            <h5>Best Parameters (Score: {bestScore})</h5>
            <table className="params-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(bestParameters).map(([name, value]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {tuningHistory.length > 0 && (
          <div className="tuning-history">
            <h5>Tuning History</h5>
            <div className="history-chart">
              {tuningHistory.map((point, i) => (
                <div 
                  key={i} 
                  className="history-point" 
                  style={{
                    left: `${(i / Math.max(iterations, tuningHistory.length)) * 100}%`,
                    bottom: `${point.score * 100}%`
                  }}
                  title={`Iteration ${point.iteration}: ${point.score}`}
                />
              ))}
            </div>
          </div>
        )}
        
        {tuningLogs.length > 0 && (
          <div className="tuning-logs">
            <h5>Tuning Logs</h5>
            <div className="logs-container">
              {tuningLogs.map((log, index) => (
                <div key={index} className="log-entry">{log}</div>
              ))}
            </div>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default ParameterTuningNode; 