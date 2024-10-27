import { useCallback, useEffect, useState } from 'react';
import { Handle, Position } from 'reactflow';

const handleStyle = { left: 10 };

function TextUpdaterNode({ data, isConnectable }) {
  // Local state to hold the text value of the input
  const [text, setText] = useState(data.label || ''); // Initialize with the label from props

  // Update local state when props change (for example, when loading nodes)
  useEffect(() => {
    setText(data.label || ''); // Ensure the input reflects the current data
  }, [data.label]);

  const onChange = useCallback((evt) => {
    const newText = evt.target.value;
    setText(newText); // Update local state
    // Optionally, if you want to notify the parent or perform actions on change
    // For example: updateNode(data.id, newText);
    console.log(newText);
  }, []);

  return (
    <div className="text-updater-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="grid-container">
        <label>Node ID:</label> 
        <span>{data.id}</span> {/* Accessing node id from data */}
        <label htmlFor="text">Text:</label>
        <input 
          id="text" 
          name="text" 
          value={text} // Bind the input's value to local state
          onChange={onChange} // Update local state on change
          className="nodrag" 
        />
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
      />
    </div>
  );
}

export default TextUpdaterNode;
