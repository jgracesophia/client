import requests

# Define the server URL
SERVER_URL = 'http://your-server-ip:5000/update_model'

# Example client data (weights and biases)
client_weights = [0.1, 0.2, 0.3]  # Example weights
client_biases = [0.01, 0.02]       # Example biases

# Send model parameters to the server for aggregation
def send_model_parameters(weights, biases):
    data = {'weights': weights, 'biases': biases}
    response = requests.post(SERVER_URL, json=data)
    if response.status_code == 200:
        updated_model = response.json()
        return updated_model
    else:
        print("Failed to update model on the server. Status code:", response.status_code)

if __name__ == '__main__':
    # Example training loop
    for i in range(5):  # Perform 5 training rounds
        # Perform local training...
        
        # After local training, send model parameters to the server
        updated_model = send_model_parameters(client_weights, client_biases)
        
        # Update client model with the aggregated model from the server
        if updated_model:
            client_weights = updated_model['weights']
            client_biases = updated_model['biases']
            print("Round", i+1, "Model updated successfully:", client_weights, client_biases)
