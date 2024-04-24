import os
import csv

def save_labels_to_csv(root_dir, csv_filename):
    # List all directories within the root directory
    labels = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    
    # Save the labels to a CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label'])  # Writing header
        for label in labels:
            writer.writerow([label])  # Writing each label as a new row

# Example usage
root_dir = '../preprocessed'  # Path to your dataset directory
csv_filename = 'labels.csv'  # Desired CSV filename
save_labels_to_csv(root_dir, csv_filename)
print(f"Labels have been saved to {csv_filename}")
