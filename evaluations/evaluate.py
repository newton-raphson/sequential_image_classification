# from executor.executor import Executor
from model.networks import AutoEncoder,SequentialModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_autoencoder(model, post_process_path, device, test_dataloader):
    # Load the pre-trained autoencoder model
    # model = Executor.load_model_ae(model, model_path)
    
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device

    # Initialize variables to accumulate reconstruction error
    mse_error = 0.0
    num_batches = len(test_dataloader)

    # Iterate over the test data loader
    for inputs, _ in test_dataloader:
        # Move inputs to the specified device
        inputs = inputs.to(device)
        
        # Forward pass to get reconstructed outputs
        with torch.no_grad():  # No need to track gradients during evaluation
            reconstructed_outputs = model(inputs)
        
        # Compute mean squared error for this batch
        batch_mse_error = torch.mean((inputs - reconstructed_outputs) ** 2).item()
        
        # Accumulate batch MSE error
        mse_error += batch_mse_error

    # Compute average MSE error across all batches
    mse_error /= num_batches

    # Save the MSE error in a text file
    mse_file_path = os.path.join(post_process_path, 'mse_error.txt')
    with open(mse_file_path, 'w') as file:
        file.write(f'MSE Error: {mse_error}')

    # Save true and reconstructed images side by side
    for i, (true_image, reconstructed_image) in enumerate(zip(test_dataloader.dataset, reconstructed_outputs)):
        # Convert true and reconstructed images to numpy arrays
        true_image_np = true_image[0].squeeze().cpu().numpy()
        reconstructed_image_np = reconstructed_image.squeeze().cpu().numpy()

        # Create a combined image with true and reconstructed images side by side
        combined_image = np.concatenate([true_image_np, reconstructed_image_np], axis=1)

        # Save the combined image
        image_file_path = os.path.join(post_process_path, f'image_pair_{i}.png')
        plt.imsave(image_file_path, combined_image, cmap='gray')

    return mse_error



def evaluate_sequential(model, autoencoder, post_process_path, device, test_dataloader):
    model.to(device=device)
    model.eval()
    # Load the autoencoder and extract the encoder

    encoder = autoencoder.encoder.to(device)
    encoder.eval()  # Set the encoder to evaluation mode

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over the test data loader
    for inputs, labels in test_dataloader:
        # Move inputs to the specified device
        inputs = inputs.to(device)
        
        # Forward pass through the encoder
        encoded_inputs = encoder(inputs)
        
        # Forward pass through the sequential model
        outputs = model(encoded_inputs)
        
        # Convert probabilities to predicted labels (0 or 1)
        predicted = torch.round(outputs).squeeze().detach().cpu().numpy()
        
        # Append true labels and predicted labels
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted)

    # Compute classification accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Compute confusion matrix
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    
    # Compute classification report
    class_report = classification_report(true_labels, predicted_labels)
    
    # Save the accuracy to a text file
    accuracy_file_path = os.path.join(post_process_path, 'accuracy.txt')
    with open(accuracy_file_path, 'w') as file:
        file.write(f'Accuracy: {accuracy}')

    # Save the confusion matrix to a text file
    confusion_mat_file_path = os.path.join(post_process_path, 'confusion_matrix.txt')
    with open(confusion_mat_file_path, 'w') as file:
        file.write('Confusion Matrix:\n')
        file.write(str(confusion_mat))

    # Save the classification report to a text file
    class_report_file_path = os.path.join(post_process_path, 'classification_report.txt')
    with open(class_report_file_path, 'w') as file:
        file.write('Classification Report:\n')
        file.write(class_report)

    return accuracy, confusion_mat, class_report