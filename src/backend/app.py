from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# In-memory storage for node data (this can be replaced with a database)
node_storage = {}

@app.route('/save_nodes', methods=['POST'])
def save_nodes():
    try:
        data = request.json
        # Assuming the request body contains `nodes` and `filename`
        filename = data.get("filename", "default_nodes.json")
        nodes = data.get("nodes", [])
        
        # Save nodes to in-memory storage or file (if needed)
        node_storage[filename] = nodes

        # Save nodes to a file if needed
        with open(filename, 'w') as f:
            json.dump(nodes, f)

        return jsonify({"message": "Nodes saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_nodes', methods=['GET'])
def load_nodes():
    try:
        # Retrieve the filename from query parameters
        filename = request.args.get("filename", "default_nodes.json")

        if filename in node_storage:
            nodes = node_storage[filename]
        else:
            # Load from file if in-memory storage does not have the data
            with open(filename, 'r') as f:
                nodes = json.load(f)

        return jsonify({"nodes": nodes}), 200
    except FileNotFoundError:
        return jsonify({"error": f"File '{filename}' not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
