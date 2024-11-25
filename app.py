from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import logging
from neural_networks import visualize

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

logging.basicConfig(level=logging.DEBUG)
# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        app.logger.debug("Request received at /run_experiment")

        # Parse JSON data from the request
        data = request.get_json()
        if data is None:
            app.logger.error("Invalid JSON in request.")
            return jsonify({"error": "Invalid JSON in request."}), 400

        activation = data.get('activation')
        lr = float(data.get('lr', 0.1))
        step_num = int(data.get('step_num', 1000))

        app.logger.debug(f"Parsed data - Activation: {activation}, Learning Rate: {lr}, Steps: {step_num}")

        # Validate inputs
        if activation not in ['relu', 'tanh', 'sigmoid']:
            app.logger.error("Invalid activation function.")
            return jsonify({"error": "Invalid activation function. Choose from 'relu', 'tanh', 'sigmoid'."}), 400

        if lr <= 0:
            app.logger.error("Learning rate must be positive.")
            return jsonify({"error": "Learning rate must be positive."}), 400

        if step_num <= 0:
            app.logger.error("Number of training steps must be positive.")
            return jsonify({"error": "Number of training steps must be a positive integer."}), 400

        # Run the visualization function
        app.logger.debug("Running visualization function...")
        visualize(activation, lr, step_num)
        app.logger.debug("Visualization completed.")

        # Check if the result GIF exists
        result_gif = "results/visualize.gif"
        if os.path.exists(result_gif):
            app.logger.debug(f"Visualization file found at {result_gif}.")
            # Append a unique query parameter (e.g., timestamp) to prevent caching
            return jsonify({"result_gif": f"{result_gif}?t={int(os.path.getmtime(result_gif))}"}), 200
        else:
            app.logger.error("Visualization file not generated.")
            return jsonify({"error": "Visualization file not generated."}), 500

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)