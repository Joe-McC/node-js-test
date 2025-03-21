import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import './styles/node-styles.css';

function ModelTrainingNode({ data, isConnectable }) {
  const [modelType, setModelType] = useState(data.modelType || 'random_forest');
  const [framework, setFramework] = useState(data.framework || 'scikit-learn');
  const [datasetInfo, setDatasetInfo] = useState(data.datasetInfo || null);
  const [modelParams, setModelParams] = useState(data.modelParams || {
    random_forest: {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 2
    },
    svm: {
      C: 1.0,
      kernel: 'rbf',
      gamma: 'scale'
    },
    neural_network: {
      hidden_layer_sizes: '100,50',
      activation: 'relu',
      max_iter: 200
    }
  });
  const [trainingLogs, setTrainingLogs] = useState(data.trainingLogs || []);
  const [status, setStatus] = useState(data.status || 'Not Started');
  const [progress, setProgress] = useState(data.progress || 0);
  const [metrics, setMetrics] = useState(data.metrics || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Update parent data when state changes
  useEffect(() => {
    if (data.onChange) {
      data.onChange({
        ...data,
        modelType,
        framework,
        datasetInfo,
        modelParams,
        trainingLogs,
        status,
        progress,
        metrics
      });
    }
  }, [data.onChange, modelType, framework, trainingLogs, status, progress, metrics]);

  // Update dataset info if it's passed in from another node
  useEffect(() => {
    // Only update if there's actual new data and it's different
    console.log('ModelTrainingNode: Checking for dataset updates');
    console.log('ModelTrainingNode: Current datasetInfo:', datasetInfo);
    console.log('ModelTrainingNode: Incoming datasetInfo:', data.datasetInfo);
    console.log('ModelTrainingNode: Incoming dataset name:', data.dataset);
    
    // Handle case where dataset and datasetInfo conflict - prioritize direct dataset name
    if (data.dataset && data.datasetInfo && data.dataset !== data.datasetInfo.name) {
      console.log('ModelTrainingNode: Dataset name conflicts with datasetInfo - using dataset name:', data.dataset);
      setDatasetInfo({
        name: data.dataset,
        splitRatio: data.datasetInfo?.splitRatio || 0.8,
        stats: data.datasetInfo?.stats || {},
        timestamp: new Date().getTime(),
        source: 'direct'
      });
      return;
    }
    
    // Direct dataset name property - highest priority
    if (data.dataset && (!datasetInfo || data.dataset !== datasetInfo.name)) {
      console.log('ModelTrainingNode: Setting dataset from direct name:', data.dataset);
      setDatasetInfo({
        name: data.dataset,
        splitRatio: datasetInfo?.splitRatio || 0.8,
        stats: datasetInfo?.stats || {},
        timestamp: new Date().getTime(),
        source: 'direct'
      });
      return;
    }
    
    // datasetInfo object - second priority
    if (data.datasetInfo && data.datasetInfo.name) {
      // Use the timestamp to detect changes, or compare name if no timestamp
      const dataChanged = !datasetInfo || 
                          data.datasetInfo.timestamp !== datasetInfo.timestamp || 
                          data.datasetInfo.name !== datasetInfo.name ||
                          data.datasetInfo.uuid !== datasetInfo.uuid;
      
      if (dataChanged) {
        console.log('ModelTrainingNode: Setting dataset from datasetInfo:', data.datasetInfo);
        setDatasetInfo({
          ...data.datasetInfo,
          source: 'info'
        });
      }
    }
  }, [data.datasetInfo, data.dataset]);

  // Handle model parameters changes in a separate effect
  useEffect(() => {
    if (data.modelParams) {
      // Create a copy to ensure we don't create an infinite update loop
      const paramsJSON = JSON.stringify(data.modelParams);
      const currentParamsJSON = JSON.stringify(modelParams);
      
      if (paramsJSON !== currentParamsJSON) {
        console.log('ModelTrainingNode: Received new model parameters');
        const updatedParams = { ...modelParams };
        
        if (data.modelParams.n_estimators) {
          updatedParams.random_forest = { 
            ...updatedParams.random_forest, 
            ...data.modelParams 
          };
        } else if (data.modelParams.C) {
          updatedParams.svm = { 
            ...updatedParams.svm, 
            ...data.modelParams 
          };
        } else if (data.modelParams.hidden_layer_sizes) {
          updatedParams.neural_network = { 
            ...updatedParams.neural_network, 
            ...data.modelParams 
          };
        }
        
        setModelParams(updatedParams);
      }
    }
  }, [data.modelParams]);

  const handleParamChange = (model, param, value) => {
    setModelParams({
      ...modelParams,
      [model]: {
        ...modelParams[model],
        [param]: value
      }
    });
  };

  const trainModel = async () => {
    // Force a dataset check before proceeding
    const finalDatasetName = data.dataset || 
                            (data.datasetInfo && data.datasetInfo.name) || 
                            (datasetInfo && datasetInfo.name);
    
    if (!finalDatasetName) {
      setError('No dataset information available. Connect to a Data Preparation node first.');
      return;
    }

    // If current state doesn't match incoming props, update it first
    if (data.dataset && (!datasetInfo || data.dataset !== datasetInfo.name)) {
      console.log('ModelTrainingNode: Updating dataset before training to:', data.dataset);
      setDatasetInfo({
        name: data.dataset,
        timestamp: new Date().getTime(),
        source: 'immediate_update'
      });
      
      // Short delay to let state update
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setLoading(true);
    setError(null);
    setStatus('Training');
    setProgress(0);
    
    // Reset logs and metrics before starting
    setTrainingLogs([]);
    setMetrics(null);
    
    // Log dataset information for debugging
    console.log(`ModelTrainingNode: Training with dataset:`, finalDatasetName);
    
    // Add initial message
    setTrainingLogs([
      `[${new Date().toLocaleTimeString()}] Starting model training...`,
      `[${new Date().toLocaleTimeString()}] Using dataset: ${finalDatasetName}`
    ]);
    
    console.log(`ModelTrainingNode: Starting to train model with dataset:`, {
      datasetName: finalDatasetName,
      modelType,
      params: modelParams[modelType]
    });
    
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 10;
        return newProgress > 95 ? 95 : newProgress;
      });
    }, 500);
    
    try {
      // Prepare model parameters based on selected model type
      const selectedParams = modelParams[modelType];
      
      // Prepare request data
      const requestData = {
        datasetName: finalDatasetName,
        modelType,
        framework,
        params: selectedParams
      };
      
      console.log('ModelTrainingNode: Sending API request with data:', requestData);
      
      // Call API to train model
      const response = await fetch('http://localhost:5000/api/train_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to train model');
      }
      
      const result = await response.json();
      console.log('ModelTrainingNode: Received training result:', result);
      
      clearInterval(progressInterval);
      setProgress(100);
      setStatus('Trained');
      setMetrics(result.metrics);
      
      // Add logs
      setTrainingLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] Training completed.`,
        `[${new Date().toLocaleTimeString()}] Accuracy: ${result.metrics.accuracy.toFixed(4)}`,
        `[${new Date().toLocaleTimeString()}] Model saved as ${result.modelId || 'unknown'}`
      ]);
      
    } catch (err) {
      clearInterval(progressInterval);
      setError(err.message);
      setStatus('Error');
      setTrainingLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] Error: ${err.message}`
      ]);
    } finally {
      setLoading(false);
    }
  };

  const renderModelParams = () => {
    // Initialize default params if they don't exist
    if (!modelParams) {
      return <div className="error-message">Model parameters not initialized</div>;
    }

    // Ensure the model type exists in modelParams
    if (!modelParams[modelType]) {
      // Initialize the missing model type with default values
      const defaultParams = {
        random_forest: {
          n_estimators: 100,
          max_depth: 10,
          min_samples_split: 2
        },
        svm: {
          C: 1.0,
          kernel: 'rbf',
          gamma: 'scale'
        },
        neural_network: {
          hidden_layer_sizes: '100,50',
          activation: 'relu',
          max_iter: 200
        }
      };

      // Update modelParams with defaults
      setModelParams({
        ...modelParams,
        ...defaultParams
      });

      // Return a loading message
      return <div>Loading parameters...</div>;
    }

    switch (modelType) {
      case 'random_forest':
        return (
          <>
            <div className="param-row">
              <label>n_estimators:</label>
              <input
                type="number"
                min="10"
                max="1000"
                value={modelParams.random_forest.n_estimators || 100}
                onChange={(e) => handleParamChange('random_forest', 'n_estimators', parseInt(e.target.value))}
              />
            </div>
            <div className="param-row">
              <label>max_depth:</label>
              <input
                type="number"
                min="1"
                max="100"
                value={modelParams.random_forest.max_depth || 10}
                onChange={(e) => handleParamChange('random_forest', 'max_depth', parseInt(e.target.value))}
              />
            </div>
            <div className="param-row">
              <label>min_samples_split:</label>
              <input
                type="number"
                min="2"
                max="20"
                value={modelParams.random_forest.min_samples_split || 2}
                onChange={(e) => handleParamChange('random_forest', 'min_samples_split', parseInt(e.target.value))}
              />
            </div>
          </>
        );
      
      case 'svm':
        return (
          <>
            <div className="param-row">
              <label>C:</label>
              <input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                value={modelParams.svm.C || 1.0}
                onChange={(e) => handleParamChange('svm', 'C', parseFloat(e.target.value))}
              />
            </div>
            <div className="param-row">
              <label>kernel:</label>
              <select
                value={modelParams.svm.kernel || 'rbf'}
                onChange={(e) => handleParamChange('svm', 'kernel', e.target.value)}
              >
                <option value="linear">linear</option>
                <option value="poly">poly</option>
                <option value="rbf">rbf</option>
                <option value="sigmoid">sigmoid</option>
              </select>
            </div>
            <div className="param-row">
              <label>gamma:</label>
              <select
                value={modelParams.svm.gamma || 'scale'}
                onChange={(e) => handleParamChange('svm', 'gamma', e.target.value)}
              >
                <option value="scale">scale</option>
                <option value="auto">auto</option>
              </select>
            </div>
          </>
        );
      
      case 'neural_network':
        return (
          <>
            <div className="param-row">
              <label>hidden_layer_sizes:</label>
              <input
                type="text"
                value={modelParams.neural_network.hidden_layer_sizes || '100,50'}
                onChange={(e) => handleParamChange('neural_network', 'hidden_layer_sizes', e.target.value)}
                placeholder="comma separated, e.g. 100,50"
              />
            </div>
            <div className="param-row">
              <label>activation:</label>
              <select
                value={modelParams.neural_network.activation || 'relu'}
                onChange={(e) => handleParamChange('neural_network', 'activation', e.target.value)}
              >
                <option value="relu">ReLU</option>
                <option value="tanh">tanh</option>
                <option value="sigmoid">sigmoid</option>
              </select>
            </div>
            <div className="param-row">
              <label>max_iter:</label>
              <input
                type="number"
                min="100"
                max="1000"
                value={modelParams.neural_network.max_iter || 200}
                onChange={(e) => handleParamChange('neural_network', 'max_iter', parseInt(e.target.value))}
              />
            </div>
          </>
        );
    
      default:
        return <div>Select a model type to view parameters</div>;
    }
  };

  return (
    <div className="ml-node model-training-node">
      <div className="node-header model-train">
        <div>Model Training</div>
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
        {datasetInfo ? (
          <div className="dataset-info">
            <h5>Dataset: {typeof datasetInfo === 'string' ? datasetInfo : datasetInfo.name}</h5>
            <p>Train/Test split: {typeof datasetInfo === 'object' && datasetInfo.splitRatio ? datasetInfo.splitRatio * 100 : 80}%</p>
            {typeof datasetInfo === 'object' && datasetInfo.stats && (
              <p>Rows: {datasetInfo.stats.rows}, Features: {datasetInfo.stats.columns}</p>
            )}
          </div>
        ) : (
          <div className="no-dataset">
            <p>No dataset connected. Connect to a Data Preparation node.</p>
          </div>
        )}
        
        <div className="input-group">
          <label>Model Type:</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
          >
            <option value="random_forest">Random Forest</option>
            <option value="svm">SVM</option>
            <option value="neural_network">Neural Network</option>
          </select>
        </div>
        
        <div className="input-group">
          <label>Framework:</label>
          <select
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
          >
            <option value="scikit-learn">scikit-learn</option>
            <option value="tensorflow">TensorFlow</option>
            <option value="pytorch">PyTorch</option>
          </select>
        </div>
        
        <div className="training-params">
          <h5>Model Parameters</h5>
          {renderModelParams()}
        </div>
        
        <button 
          className="ml-node-button train-button"
          onClick={trainModel}
          disabled={loading || !datasetInfo}
        >
          {loading ? 'Training...' : 'Train Model'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
        
        <div className="status">Status: {status}</div>
        
        {status === 'Training' && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}>
              <span>{Math.round(progress)}%</span>
            </div>
          </div>
        )}
        
        {metrics && (
          <div className="metrics-container">
            <h5>Training Results</h5>
            <div className="metric-display">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="metric-item">
                  <span>{key}:</span>
                  <span>{typeof value === 'number' ? value.toFixed(4) : value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {trainingLogs.length > 0 && (
          <div className="logs-section">
            <h5>Training Logs</h5>
            <div className="logs-container">
              {trainingLogs.map((log, index) => (
                <div key={index} className="log-entry">{log}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelTrainingNode; 