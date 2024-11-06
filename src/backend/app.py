from flask import Flask, request, jsonify
import json
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for node data (this can be replaced with a database)
node_storage = {}

@app.route('/save_nodes', methods=['POST'])
def save_nodes():
    try:
        data = request.json
        print(f"Received data: {data}")  # Debugging print

        filename = data.get("filename", "default_nodes.json")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Check if nodes and filename are valid
        if not filename or not nodes:
            print("Filename or nodes data missing")  # Debugging print
            return jsonify({"error": "Filename or nodes data missing."}), 400

        # Save nodes to in-memory storage
        node_storage[filename] = nodes

        # Save nodes to file on disk
        with open(filename, 'w') as f:
            json.dump({"nodes": nodes, "edges": edges}, f)

        print(f"Nodes successfully saved to {filename}")  # Debugging print
        return jsonify({"message": "Nodes saved successfully!"}), 200
    except Exception as e:
        print(f"Error saving nodes: {str(e)}")  # Debugging print
        return jsonify({"error": str(e)}), 500


@app.route('/load_nodes', methods=['GET'])
def load_nodes():
    try:
        filename = request.args.get("filename", "default_nodes.json")

        # Check if file exists in in-memory storage
        if filename in node_storage:
            data = node_storage[filename]
        else:
            # Check if file exists on disk
            with open(filename, 'r') as f:
                data = json.load(f)
        return jsonify(data)
        #return jsonify({"nodes": nodes}), 200
    except FileNotFoundError:
        return jsonify({"error": f"File '{filename}' not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
