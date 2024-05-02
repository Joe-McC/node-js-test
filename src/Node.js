// Node.js
import React from 'react';

export default function Node({ data, id }) {
  const handleInputChange = (event) => {
    data.label = event.target.value;
  };

  return (
    <div>
      <input type="text" defaultValue={data.label} onChange={handleInputChange} />
    </div>
  );
}

