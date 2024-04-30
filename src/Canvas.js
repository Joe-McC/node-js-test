// Canvas.js
import React from 'react';
import { useDrop } from 'react-dnd';

const Canvas = ({ onDrop, children }) => {
  const [{ isOver }, drop] = useDrop({
    accept: 'node',
    drop: (item, monitor) => {
      const delta = monitor.getDifferenceFromInitialOffset();
      const left = Math.round(item.left + delta.x);
      const top = Math.round(item.top + delta.y);
      onDrop(item.id, left, top);
    },
    collect: (monitor) => ({
      isOver: !!monitor.isOver(),
    }),
  });

  return (
    <div ref={drop} style={{ position: 'relative', width: '100%', height: '100%' }}>
      {children}
    </div>
  );
};

export default Canvas;