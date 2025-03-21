import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import './styles/node-styles.css';

function DataPrepNode({ data, isConnectable }) {
  const [datasetName, setDatasetName] = useState(data.datasetName || '');
  const [splitRatio, setSplitRatio] = useState(data.splitRatio || 0.8);
  const [preprocessingSteps, setPreprocessingSteps] = useState(data.preprocessingSteps || []);
  const [newStep, setNewStep] = useState('');
  const [isConfigured, setIsConfigured] = useState(data.isConfigured || false);
  const [datasetStats, setDatasetStats] = useState(data.datasetStats || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useExampleDataset, setUseExampleDataset] = useState(false);
  const [selectedExampleDataset, setSelectedExampleDataset] = useState('');

  // Example datasets
  const exampleDatasets = [
    { value: '', label: 'Select a dataset' },
    { value: 'iris', label: 'Iris (Classification)' },
    { value: 'boston_housing', label: 'Boston Housing (Regression)' },
    { value: 'customer_churn', label: 'Customer Churn (Classification)' },
    { value: 'wine_quality', label: 'Wine Quality (Regression)' },
    { value: 'diabetes', label: 'Diabetes (Classification)' }
  ];

  // Update parent data when dataset settings change
  useEffect(() => {
    if (data.onChange) {
      data.onChange({
        ...data,
        datasetName,
        splitRatio,
        preprocessingSteps,
        isConfigured,
        datasetStats,
        useExampleDataset,
        selectedExampleDataset
      });
    }
  }, [data, datasetName, splitRatio, preprocessingSteps, isConfigured, datasetStats, useExampleDataset, selectedExampleDataset]);

  // Handle example dataset selection
  const handleExampleDatasetChange = (e) => {
    const selected = e.target.value;
    setSelectedExampleDataset(selected);
    if (selected) {
      setDatasetName(selected);
    }
  };

  const toggleDatasetSource = () => {
    setUseExampleDataset(!useExampleDataset);
    if (!useExampleDataset) {
      setDatasetName(selectedExampleDataset);
    } else {
      setSelectedExampleDataset('');
    }
  };

  const addPreprocessingStep = () => {
    if (newStep.trim()) {
      setPreprocessingSteps([...preprocessingSteps, newStep.trim()]);
      setNewStep('');
    }
  };

  const removePreprocessingStep = (index) => {
    const updatedSteps = [...preprocessingSteps];
    updatedSteps.splice(index, 1);
    setPreprocessingSteps(updatedSteps);
  };

  const configureDataset = async () => {
    if (!datasetName) {
      setError('Dataset name is required');
      return;
    }

    console.log(`DataPrepNode: Configuring dataset: ${datasetName}`);
    setLoading(true);
    setError(null);
    
    try {
      // Call API to configure dataset
      const response = await fetch('http://localhost:5000/api/prepare_dataset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          datasetName,
          splitRatio,
          preprocessingSteps,
          useExampleDataset
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to configure dataset');
      }
      
      const result = await response.json();
      setDatasetStats(result.datasetStats);
      setIsConfigured(true);
      
      console.log(`DataPrepNode: Dataset configured successfully, name: ${datasetName}`);
      console.log(`DataPrepNode: Dataset stats:`, result.datasetStats);
      
      // Force all data to be updated together
      const updateAll = () => {
        // Create a dataset info object with timestamp to ensure change detection
        const datasetInfo = {
          name: datasetName,
          splitRatio: splitRatio,
          stats: result.datasetStats,
          nodeId: data.id,
          // Add timestamp and UUID to force recognition as new object
          timestamp: new Date().getTime(),
          uuid: Math.random().toString(36).substring(2, 15)
        };
        
        console.log('DataPrepNode: Created dataset info object:', JSON.stringify(datasetInfo));
        
        // Use the App.js callback mechanism
        if (data.onDatasetConfigured) {
          console.log('DataPrepNode: Sending dataset info via callback with name:', datasetName);
          data.onDatasetConfigured(datasetInfo);
        } else {
          console.warn('DataPrepNode: No onDatasetConfigured callback available');
        }
        
        // Also update via onChange callback for redundancy
        if (data.onChange) {
          console.log('DataPrepNode: Updating node data with dataset info');
          data.onChange({
            ...data,
            datasetInfo,
            dataset: datasetName,
            datasetName: datasetName,
            datasetConfigured: true,
            isConfigured: true
          });
        }
      };
      
      // Call with slight delay to ensure state is fully updated
      setTimeout(updateAll, 100);
      
    } catch (err) {
      setError(err.message);
      setIsConfigured(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ml-node data-prep-node">
      <div className="node-header data-processor">
        <div>Data Preparation</div>
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
        <div className="dataset-source-toggle">
          <label>
            <input
              type="checkbox"
              checked={useExampleDataset}
              onChange={toggleDatasetSource}
            />
            Use example dataset
          </label>
        </div>
        
        {useExampleDataset ? (
          <div className="input-group">
            <label>Select Example Dataset:</label>
            <select
              value={selectedExampleDataset}
              onChange={handleExampleDatasetChange}
              className="dataset-select"
            >
              {exampleDatasets.map(dataset => (
                <option key={dataset.value} value={dataset.value}>
                  {dataset.label}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="input-group">
            <label>Dataset Name/Path:</label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g., iris, boston, custom.csv"
            />
          </div>
        )}
        
        <div className="input-group">
          <label>Train/Test Split Ratio: {splitRatio}</label>
          <div className="slider-container">
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={splitRatio}
              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
            />
            <span>{(splitRatio * 100).toFixed(0)}%</span>
          </div>
        </div>
        
        <div className="preprocessing-steps">
          <h5>Preprocessing Steps</h5>
          <ul>
            {preprocessingSteps.map((step, index) => (
              <li key={index}>
                {step}
                <button 
                  className="small-button"
                  onClick={() => removePreprocessingStep(index)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
          
          <div className="input-row">
            <input
              type="text"
              value={newStep}
              onChange={(e) => setNewStep(e.target.value)}
              placeholder="e.g., normalize, drop_nulls, one_hot_encode"
            />
            <button 
              className="small-button"
              onClick={addPreprocessingStep}
              disabled={!newStep.trim()}
            >
              Add
            </button>
          </div>
        </div>
        
        <button 
          className="ml-node-button"
          onClick={configureDataset}
          disabled={loading || !datasetName}
        >
          {loading ? 'Configuring...' : 'Configure Dataset'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
        
        {isConfigured && (
          <div className="dataset-stats">
            <h5>Dataset Statistics</h5>
            {datasetStats && (
              <>
                <p>Rows: {datasetStats.rows}</p>
                <p>Columns: {datasetStats.columns}</p>
                <p>Train Set: {datasetStats.trainSize} rows</p>
                <p>Test Set: {datasetStats.testSize} rows</p>
                {datasetStats.features && (
                  <p>Features: {Array.isArray(datasetStats.features) 
                    ? datasetStats.features.join(', ') 
                    : (typeof datasetStats.features === 'string' 
                      ? datasetStats.features 
                      : JSON.stringify(datasetStats.features))}</p>
                )}
                {datasetStats.target && (
                  <p>Target: {datasetStats.target}</p>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default DataPrepNode; 