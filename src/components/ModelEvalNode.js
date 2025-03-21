import React, { useState, useEffect, useCallback } from 'react';
import { Handle, Position } from 'reactflow';
import '../styles/Nodes.css';

function ModelEvalNode({ data, isConnectable }) {
  const [metrics, setMetrics] = useState(data.metrics || [
    { name: 'Accuracy', value: 0, target: 0.9 },
    { name: 'Precision', value: 0, target: 0.85 },
    { name: 'Recall', value: 0, target: 0.8 },
    { name: 'F1 Score', value: 0, target: 0.85 },
    { name: 'Latency (ms)', value: 0, target: 100 }
  ]);
  const [confusionMatrix, setConfusionMatrix] = useState(data.confusionMatrix || []);
  const [evalStatus, setEvalStatus] = useState(data.evalStatus || 'Not Started');
  const [evalLogs, setEvalLogs] = useState(data.evalLogs || []);
  const [requirementsMet, setRequirementsMet] = useState(data.requirementsMet || false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedPlot, setSelectedPlot] = useState(null);
  const [reportUrl, setReportUrl] = useState(data.reportUrl || '');
  const [driftDetected, setDriftDetected] = useState(data.driftDetected || false);
  const [selectedMetrics, setSelectedMetrics] = useState(data.selectedMetrics || [
    'ClassificationQuality', 'DataQuality'
  ]);
  const [dataset, setDataset] = useState(data.dataset || '');
  const [modelType, setModelType] = useState(data.modelType || '');
  
  const allPossibleMetrics = [
    'ClassificationQuality', 'RegressionQuality', 'DataQuality', 
    'ProbClassificationQuality', 'DataDrift'
  ];

  useEffect(() => {
    // Update node data when state changes
    if (data.onChange) {
      data.onChange({
        ...data,
        metrics,
        confusionMatrix,
        evalStatus,
        evalLogs,
        requirementsMet,
        selectedMetrics,
        evaluationResults: metrics,
        reportUrl,
        driftDetected
      });
    }
  }, [metrics, confusionMatrix, evalStatus, evalLogs, requirementsMet, selectedMetrics, reportUrl, driftDetected, data]);

  useEffect(() => {
    if (data.dataset && data.dataset !== dataset) {
      setDataset(data.dataset);
      console.log(`ModelEvalNode: Dataset updated to ${data.dataset}`);
    }
  }, [data.dataset, dataset]);

  useEffect(() => {
    if (data.input && data.input.modelType && data.input.modelType !== modelType) {
      setModelType(data.input.modelType);
      console.log(`ModelEvalNode: Model type updated to ${data.input.modelType}`);
    }
  }, [data.input, modelType]);

  const handleEvaluate = useCallback(async () => {
    if (!modelType) {
      setError('Please select a model type');
      return;
    }
    
    if (!dataset) {
      setError('Please specify a dataset');
      return;
    }
    
    setLoading(true);
    setError(null);
    setEvalStatus('Evaluating');
    setEvalLogs([`[${new Date().toLocaleTimeString()}] Starting model evaluation...`]);
    
    try {
      // Call API to evaluate model
      const response = await fetch('http://localhost:5000/api/evaluate_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelType: modelType,
          dataset: dataset,
          metrics: selectedMetrics
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to evaluate model');
      }
      
      const resultData = await response.json();
      setResults(resultData);
      
      // Update metrics from the evaluation results
      if (resultData.evaluationResults) {
        setMetrics(resultData.evaluationResults.map(metric => ({
          ...metric,
          target: metrics.find(m => m.name === metric.name)?.target || 
                  (metric.name === 'Latency (ms)' ? 100 : 0.8)
        })));
      }
      
      // Set confusion matrix if available
      if (resultData.confusionMatrix) {
        setConfusionMatrix(resultData.confusionMatrix);
      }
      
      // Check if requirements are met
      const allRequirementsMet = resultData.evaluationResults ? 
        resultData.evaluationResults.every(metric => {
          if (metric.name === 'Latency (ms)') {
            return metric.value <= (metrics.find(m => m.name === metric.name)?.target || 100);
          }
          return metric.value >= (metrics.find(m => m.name === metric.name)?.target || 0.8);
        }) : false;
      
      setRequirementsMet(allRequirementsMet);
      setEvalStatus(allRequirementsMet ? 'Requirements Met' : 'Requirements Not Met');
      
      // Set report URL if available
      if (resultData.reportUrl) {
        setReportUrl(resultData.reportUrl);
      }
      
      // Set drift status
      setDriftDetected(resultData.driftDetected || false);
      
      // Add logs
      setEvalLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] Evaluation completed`,
        `[${new Date().toLocaleTimeString()}] Requirements ${allRequirementsMet ? 'satisfied' : 'not satisfied'}`,
        ...(resultData.driftDetected ? [`[${new Date().toLocaleTimeString()}] Data drift detected!`] : [])
      ]);
      
      // Update node data with results
      if (data.onResultsUpdate) {
        data.onResultsUpdate(resultData);
      }
      
    } catch (err) {
      setError(err.message);
      setEvalLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] Error: ${err.message}`
      ]);
      setEvalStatus('Error');
    } finally {
      setLoading(false);
    }
  }, [data, selectedMetrics, evalLogs, metrics, modelType, dataset]);

  const renderMetrics = (metricData) => {
    if (!metricData || !metricData.metrics || Object.keys(metricData.metrics).length === 0) {
      return <p>No metrics available</p>;
    }
    
    return (
      <div className="metric-display">
        {Object.entries(metricData.metrics).map(([key, value]) => (
          <div key={key} className="metric-item">
            <span>{key}:</span>
            <span>{typeof value === 'number' ? value.toFixed(4) : value.toString()}</span>
          </div>
        ))}
      </div>
    );
  };

  const renderPlotSelector = (metricData) => {
    if (!metricData || !metricData.plots || Object.keys(metricData.plots).length === 0) {
      return null;
    }
    
    return (
      <div className="plot-selector">
        <select 
          value={selectedPlot || ''} 
          onChange={(e) => setSelectedPlot(e.target.value)}
        >
          <option value="">Select a plot</option>
          {Object.keys(metricData.plots).map(plotKey => (
            <option key={plotKey} value={plotKey}>{plotKey}</option>
          ))}
        </select>
        
        {selectedPlot && metricData.plots[selectedPlot] && (
          <div className="plot-display">
            <img 
              src={`data:image/png;base64,${metricData.plots[selectedPlot]}`}
              alt={selectedPlot}
              style={{ maxWidth: '100%', marginTop: '10px' }}
            />
          </div>
        )}
      </div>
    );
  };

  const toggleMetricSelection = (metricName) => {
    setSelectedMetrics(prev => 
      prev.includes(metricName)
        ? prev.filter(m => m !== metricName)
        : [...prev, metricName]
    );
  };

  const viewFullReport = () => {
    if (reportUrl) {
      window.open(`http://localhost:5000${reportUrl}`, '_blank');
    }
  };

  return (
    <div className="ml-node model-eval-node">
      <div className="node-header model-eval">
        <div>Model Evaluation</div>
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
          <label>Target Column:</label>
          <input
            type="text"
            value={data.targetColumn || ''}
            onChange={(e) => {
              if (data.onChange) {
                data.onChange({ ...data, targetColumn: e.target.value });
              }
            }}
          />
        </div>
        
        <div className="metric-selector">
          <h5>Select Evidently Metrics</h5>
          <div className="metric-checkboxes">
            {allPossibleMetrics.map(metric => (
              <label key={metric} className="metric-checkbox">
                <input
                  type="checkbox"
                  checked={selectedMetrics.includes(metric)}
                  onChange={() => toggleMetricSelection(metric)}
                  className="nodrag"
                />
                {metric}
              </label>
            ))}
          </div>
        </div>
        
        <button 
          className="ml-node-button"
          onClick={handleEvaluate} 
          disabled={loading || !data.referenceData || !data.currentData}
        >
          {loading ? 'Evaluating...' : 'Evaluate Model'}
        </button>
        
        {error && <div className="error-message">{error}</div>}
        
        <div className="status">Status: {evalStatus}</div>
        {driftDetected && (
          <div className="drift-warning">⚠️ Data Drift Detected</div>
        )}
        
        {reportUrl && (
          <div className="report-link">
            <button onClick={viewFullReport} className="view-report-btn">
              View Full Evidently Report
            </button>
          </div>
        )}
        
        {metrics && metrics.length > 0 && (
          <div className="metrics-container">
            <h5>Metrics</h5>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Target</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric, index) => {
                  // Determine if the metric passes its requirement
                  const passes = metric.name === 'Latency (ms)'
                    ? metric.value <= metric.target
                    : metric.value >= metric.target;
                    
                  return (
                    <tr key={index} className={metric.value ? (passes ? 'passed' : 'failed') : ''}>
                      <td>{metric.name}</td>
                      <td>{metric.value ? (typeof metric.value === 'number' ? metric.value.toFixed(3) : metric.value) : '-'}</td>
                      <td>{metric.target}</td>
                      <td>{metric.value ? (passes ? '✓' : '✗') : '-'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {confusionMatrix && confusionMatrix.length > 0 && (
          <div className="confusion-matrix">
            <h5>Confusion Matrix</h5>
            <table className="matrix-table">
              <tbody>
                {confusionMatrix.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {results && (
          <div className="visualization-container">
            <h5>Data Quality</h5>
            {renderMetrics(results.dataQuality)}
            {renderPlotSelector(results.dataQuality)}
            
            <h5>Data Drift</h5>
            {renderMetrics(results.dataDrift)}
            {renderPlotSelector(results.dataDrift)}
            
            {results.targetDrift && (
              <>
                <h5>Target Drift</h5>
                {renderMetrics(results.targetDrift)}
                {renderPlotSelector(results.targetDrift)}
              </>
            )}
          </div>
        )}
        
        {evalLogs.length > 0 && (
          <div className="eval-logs">
            <h5>Evaluation Logs</h5>
            <div className="logs-container">
              {evalLogs.map((log, index) => (
                <div key={index} className="log-entry">{log}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelEvalNode; 