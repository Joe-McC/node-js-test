import React, { useState, useEffect, useCallback } from 'react';
import { Handle, Position } from 'reactflow';
import '../styles/Nodes.css';

function ModelTestNode({ data, isConnectable }) {
  const [testTypes, setTestTypes] = useState(data.testTypes || [
    { name: 'Data Quality Tests', selected: true },
    { name: 'Model Performance Tests', selected: true },
    { name: 'Fairness Tests', selected: false }
  ]);
  const [thresholds, setThresholds] = useState(data.thresholds || {
    accuracy: 0.8,
    precision: 0.75,
    recall: 0.7,
    f1Score: 0.75
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [testResults, setTestResults] = useState(data.testResults || null);
  const [testStatus, setTestStatus] = useState(data.testStatus || 'Not Started');
  const [dataset, setDataset] = useState(data.dataset || '');
  const [modelType, setModelType] = useState(data.modelType || 'random_forest');
  const [results, setResults] = useState(null);
  const [requirementsStatus, setRequirementsStatus] = useState(data.requirementsStatus || { passed: 0, failed: 0, total: 0 });
  const [testSet, setTestSet] = useState(data.testSet || [
    { id: 1, name: 'DataDriftTest', description: 'Test if data drift is present', passed: false },
    { id: 2, name: 'ModelPerformanceTest', description: 'Test model accuracy metrics against benchmarks', passed: false },
    { id: 3, name: 'DataQualityTest', description: 'Test if data quality meets requirements', passed: false },
    { id: 4, name: 'AdversarialTest', description: 'Test model robustness against adversarial inputs', passed: false },
    { id: 5, name: 'BiasTest', description: 'Test for bias in model predictions', passed: false }
  ]);
  const [selectedTests, setSelectedTests] = useState(data.selectedTests || testSet.map(t => t.id));
  
  // Available test datasets
  const availableDatasets = ['MNIST', 'CIFAR-10', 'Boston Housing', 'Wine Quality', 'Iris', 'Custom Dataset'];

  // Update parent data when state changes
  useEffect(() => {
    if (data.onChange) {
      data.onChange({
        ...data,
        testTypes,
        thresholds,
        testResults,
        testStatus,
        dataset,
        modelType,
        results,
        requirementsStatus,
        testSet,
        selectedTests
      });
    }
  }, [data, testTypes, thresholds, testResults, testStatus, dataset, modelType, results, requirementsStatus, testSet, selectedTests]);

  // Update dataset from props
  useEffect(() => {
    if (data.dataset && data.dataset !== dataset) {
      setDataset(data.dataset);
      console.log(`ModelTestNode: Dataset updated to ${data.dataset}`);
    }
  }, [data.dataset, dataset]);

  // Update modelType if provided from ModelTraining node
  useEffect(() => {
    if (data.input && data.input.modelType && data.input.modelType !== modelType) {
      setModelType(data.input.modelType);
      console.log(`ModelTestNode: Model type updated to ${data.input.modelType}`);
    }
  }, [data.input, modelType]);

  const toggleTestType = (index) => {
    const updatedTestTypes = [...testTypes];
    updatedTestTypes[index].selected = !updatedTestTypes[index].selected;
    setTestTypes(updatedTestTypes);
  };

  const updateThreshold = (key, value) => {
    setThresholds({
      ...thresholds,
      [key]: parseFloat(value)
    });
  };

  const handleRunTests = useCallback(async () => {
    if (!dataset) {
      setError('Please specify a dataset');
      return;
    }
    
    if (!modelType) {
      setError('Please specify a model type');
      return;
    }
    
    setLoading(true);
    setError(null);
    setTestStatus('Running Tests');
    
    try {
      // Get selected test types
      const selectedTests = testTypes
        .filter(test => test.selected)
        .map(test => test.name);
      
      if (selectedTests.length === 0) {
        throw new Error('Please select at least one test type');
      }
      
      // Call API to run tests
      const response = await fetch('http://localhost:5000/api/test_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelType: modelType,
          dataset: dataset,
          testTypes: selectedTests,
          thresholds: thresholds
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to run tests');
      }
      
      const result = await response.json();
      setTestResults(result);
      
      // Determine overall test status
      const passedAllTests = result.summary.failed === 0;
      setTestStatus(passedAllTests ? 'Passed' : 'Failed');
      
    } catch (err) {
      setError(err.message);
      setTestStatus('Error');
    } finally {
      setLoading(false);
    }
  }, [dataset, modelType, testTypes, thresholds]);

  const handleModelTypeChange = (e) => {
    const newModelType = e.target.value;
    setModelType(newModelType);
    if (data.onChange) {
      data.onChange({ ...data, modelType: newModelType });
    }
  };

  const handleTargetColumnChange = (e) => {
    const newTarget = e.target.value;
    if (data.onChange) {
      data.onChange({ ...data, targetColumn: newTarget });
    }
  };

  const toggleTestSelection = (testId) => {
    setSelectedTests(prev => 
      prev.includes(testId)
        ? prev.filter(id => id !== testId)
        : [...prev, testId]
    );
  };

  const renderMetrics = () => {
    if (!results || !results.metrics) return null;
    
    return (
      <div className="metric-display">
        {Object.entries(results.metrics).map(([key, value]) => (
          <div key={key} className="metric-item">
            <span>{key}:</span>
            <span>{typeof value === 'number' ? value.toFixed(4) : value}</span>
          </div>
        ))}
        {results.rocAuc && (
          <div className="metric-item">
            <span>ROC AUC:</span>
            <span>{results.rocAuc.toFixed(4)}</span>
          </div>
        )}
      </div>
    );
  };

  const renderRocCurve = () => {
    if (!results || !results.rocCurve) return null;
    
    // Simple ROC curve visualization using SVG
    const { fpr, tpr } = results.rocCurve;
    const svgWidth = 150;
    const svgHeight = 150;
    const padding = 20;
    
    const points = fpr.map((x, i) => {
      const xPos = padding + x * (svgWidth - 2 * padding);
      const yPos = svgHeight - (padding + tpr[i] * (svgHeight - 2 * padding));
      return `${xPos},${yPos}`;
    }).join(' ');
    
    return (
      <div className="roc-curve">
        <h5>ROC Curve</h5>
        <svg width={svgWidth} height={svgHeight}>
          {/* Diagonal line (random classifier) */}
          <line 
            x1={padding} y1={svgHeight - padding} 
            x2={svgWidth - padding} y2={padding} 
            stroke="#ccc" strokeDasharray="4" 
          />
          
          {/* Axis */}
          <line 
            x1={padding} y1={svgHeight - padding} 
            x2={svgWidth - padding} y2={svgHeight - padding} 
            stroke="#555" 
          />
          <line 
            x1={padding} y1={svgHeight - padding} 
            x2={padding} y2={padding} 
            stroke="#555" 
          />
          
          {/* ROC curve */}
          <polyline 
            points={points} 
            fill="none" 
            stroke="#2196F3" 
            strokeWidth="2" 
          />
          
          {/* Labels */}
          <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fontSize="10">
            False Positive Rate
          </text>
          <text x={5} y={svgHeight / 2} textAnchor="middle" fontSize="10" transform={`rotate(-90, 5, ${svgHeight / 2})`}>
            True Positive Rate
          </text>
        </svg>
      </div>
    );
  };

  const renderFeatureImportance = () => {
    if (!results || !results.featureImportance) return null;
    
    return (
      <div className="feature-importance">
        <h5>Feature Importance</h5>
        <img
          src={`data:image/png;base64,${results.featureImportance}`}
          alt="Feature Importance"
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>
    );
  };

  return (
    <div className="ml-node model-test-node">
      <div className="node-header model-test">
        <div>Model Testing</div>
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
            value={data.targetColumn || ''}
            onChange={handleTargetColumnChange}
          />
        </div>
        
        <div className="input-group">
          <label>Test Dataset:</label>
          <select 
            value={dataset} 
            onChange={(e) => setDataset(e.target.value)}
            className="nodrag"
          >
            <option value="">Select Test Dataset</option>
            {availableDatasets.map(dataset => (
              <option key={dataset} value={dataset}>{dataset}</option>
            ))}
          </select>
        </div>
        
        <div className="test-selection">
          <h5>Test Selection</h5>
          {testTypes.map((test, index) => (
            <div key={test.name} className="test-item">
              <label className="test-checkbox">
                <input 
                  type="checkbox"
                  checked={test.selected}
                  onChange={() => toggleTestType(index)}
                  className="nodrag"
                />
                {test.name}
              </label>
            </div>
          ))}
        </div>
        
        <button 
          className="ml-node-button"
          onClick={handleRunTests} 
          disabled={loading || !dataset || !modelType}
        >
          {loading ? 'Running Tests...' : 'Run Model Tests'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
        
        <div className="status">Status: {testStatus}</div>
        
        {testResults && (
          <div className="visualization-container">
            <h5>Test Results</h5>
            <div className="test-results">
              <div className="requirements-summary">
                <div className="summary-item passed">
                  <span>Passed: {testResults.summary?.passed || requirementsStatus.passed}</span>
                </div>
                <div className="summary-item failed">
                  <span>Failed: {testResults.summary?.failed || requirementsStatus.failed}</span>
                </div>
                <div className="summary-item total">
                  <span>Total: {testResults.summary?.total || requirementsStatus.total}</span>
                </div>
              </div>
              
              <div className="results-list">
                {testResults.testResults && testResults.testResults.map((result, index) => (
                  <div key={index} className={`result-item ${result.passed ? 'passed' : 'failed'}`}>
                    <div className="result-header">
                      <span>{result.name}</span>
                      <span>{result.passed ? '✓' : '✗'}</span>
                    </div>
                    {!result.passed && (
                      <div className="result-details">
                        {result.details}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            {renderMetrics()}
            {renderRocCurve()}
            {renderFeatureImportance()}
          </div>
        )}
        
        {testResults && (
          <div className="requirements-summary">
            <div className="summary-item passed">
              <span>Passed: {testResults.summary?.passed || requirementsStatus.passed}</span>
            </div>
            <div className="summary-item failed">
              <span>Failed: {testResults.summary?.failed || requirementsStatus.failed}</span>
            </div>
            <div className="summary-item total">
              <span>Total: {testResults.summary?.total || requirementsStatus.total}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelTestNode; 