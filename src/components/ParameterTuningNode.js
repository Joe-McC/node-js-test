import React, { useState, useCallback, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import '../styles/Nodes.css';

function ParameterTuningNode({ data, isConnectable }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [modelType, setModelType] = useState(data.modelType || 'random_forest');
  const [targetColumn, setTargetColumn] = useState(data.targetColumn || '');
  const [parameters, setParameters] = useState(data.parameters || {
    n_estimators: { min: 10, max: 100, step: 10 },
    max_depth: { min: 3, max: 10, step: 1 },
    min_samples_split: { min: 2, max: 10, step: 2 }
  });
  const [datasetName, setDatasetName] = useState(data.datasetName || '');

  // Update datasetName if it's received from parent
  useEffect(() => {
    if (data.datasetName && data.datasetName !== datasetName) {
      setDatasetName(data.datasetName);
    }
  }, [data.datasetName, datasetName]);

  const handleTuneParameters = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      if (!datasetName) {
        throw new Error('Dataset is required. Connect to a Data Preparation node first.');
      }
      
      console.log(`Starting parameter tuning for dataset: ${datasetName}, model type: ${modelType}`);
      
      // Create parameter grid for selected model
      let paramGrid = {};
      
      if (modelType === 'random_forest') {
        paramGrid = {
          n_estimators: Array.from(
            { length: Math.floor((parameters.n_estimators.max - parameters.n_estimators.min) / parameters.n_estimators.step) + 1 },
            (_, i) => parameters.n_estimators.min + i * parameters.n_estimators.step
          ),
          max_depth: Array.from(
            { length: Math.floor((parameters.max_depth.max - parameters.max_depth.min) / parameters.max_depth.step) + 1 },
            (_, i) => parameters.max_depth.min + i * parameters.max_depth.step
          ),
          min_samples_split: Array.from(
            { length: Math.floor((parameters.min_samples_split.max - parameters.min_samples_split.min) / parameters.min_samples_split.step) + 1 },
            (_, i) => parameters.min_samples_split.min + i * parameters.min_samples_split.step
          )
        };
      } else if (modelType === 'svm') {
        // Add SVM parameter grid
        paramGrid = {
          C: [0.1, 1, 10],
          kernel: ['linear', 'rbf']
        };
      } else if (modelType === 'neural_network') {
        // Add neural network parameter grid
        paramGrid = {
          hidden_layer_sizes: [[50], [100], [50, 50]],
          activation: ['relu', 'tanh'],
          alpha: [0.0001, 0.001, 0.01]
        };
      }
      
      console.log(`Parameter grid for ${modelType}:`, paramGrid);
      console.log(`Making API request with dataset: ${datasetName}`);
      
      // Call API to tune parameters
      const response = await fetch('http://localhost:5000/api/parameter-tuning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          datasetName: datasetName,
          modelType: modelType,
          paramGrid: paramGrid
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to tune parameters');
      }
      
      const resultData = await response.json();
      console.log('Parameter tuning results:', resultData);
      setResults(resultData);
      
      // Update node data with results
      if (data.onResultsUpdate) {
        data.onResultsUpdate(resultData);
      }
      
    } catch (err) {
      console.error('Parameter tuning error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [data, modelType, parameters, datasetName]);

  const handleModelTypeChange = (e) => {
    const newModelType = e.target.value;
    setModelType(newModelType);
    if (data.onChange) {
      data.onChange({ ...data, modelType: newModelType });
    }
  };

  const handleTargetColumnChange = (e) => {
    const newTarget = e.target.value;
    setTargetColumn(newTarget);
    if (data.onChange) {
      data.onChange({ ...data, targetColumn: newTarget });
    }
  };

  const handleParameterChange = (paramName, field, value) => {
    const updatedParams = {
      ...parameters,
      [paramName]: {
        ...parameters[paramName],
        [field]: value
      }
    };
    setParameters(updatedParams);
    if (data.onChange) {
      data.onChange({ ...data, parameters: updatedParams });
    }
  };

  return (
    <div className="ml-node parameter-tuning-node">
      <div className="node-header parameter-tuning">
        <div>Parameter Tuning</div>
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: '#555', width: '8px', height: '8px' }}
          isConnectable={isConnectable}
        />
        <Handle
          type="source"
          position={Position.Right}
          style={{ background: '#555', width: '8px', height: '8px' }}
          isConnectable={isConnectable}
        />
      </div>
      
      <div className="node-content">
        <div className="input-group">
          <label>Model Type:</label>
          <select value={modelType} onChange={handleModelTypeChange}>
            <option value="random_forest">Random Forest</option>
            <option value="svm">SVM</option>
            <option value="neural_network">Neural Network</option>
          </select>
        </div>
        
        <div className="input-group">
          <label>Target Column:</label>
          <input
            type="text"
            value={targetColumn}
            onChange={handleTargetColumnChange}
          />
        </div>
        
        <h5>Parameters to Tune</h5>
        <div className="parameter-list">
          {Object.keys(parameters).map(paramName => (
            <div key={paramName} className="parameter-item">
              <div className="param-name">{paramName}</div>
              <div className="param-range">
                <div className="param-row">
                  <label>Min:</label>
                  <input
                    type="number"
                    value={parameters[paramName].min}
                    onChange={(e) => handleParameterChange(paramName, 'min', Number(e.target.value))}
                  />
                </div>
                <div className="param-row">
                  <label>Max:</label>
                  <input
                    type="number"
                    value={parameters[paramName].max}
                    onChange={(e) => handleParameterChange(paramName, 'max', Number(e.target.value))}
                  />
                </div>
                <div className="param-row">
                  <label>Step:</label>
                  <input
                    type="number"
                    value={parameters[paramName].step}
                    onChange={(e) => handleParameterChange(paramName, 'step', Number(e.target.value))}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <button 
          className="ml-node-button"
          onClick={handleTuneParameters} 
          disabled={loading || !datasetName}
        >
          {loading ? 'Tuning...' : 'Tune Parameters'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
        
        {results && (
          <div className="visualization-container">
            <h5>Best Parameters</h5>
            <div className="params-table">
              <table>
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.bestParameters || {}).map(([param, value]) => (
                    <tr key={param}>
                      <td>{param}</td>
                      <td>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <h5>Performance</h5>
            <div className="metric-display">
              <div className="metric-item">
                <span>Score:</span>
                <span>{results.score ? results.score.toFixed(4) : 'N/A'}</span>
              </div>
              <div className="metric-item">
                <span>Cross-val Score:</span>
                <span>{results.crossValScore ? results.crossValScore.toFixed(4) : 'N/A'}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ParameterTuningNode; 