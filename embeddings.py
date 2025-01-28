import torch
from transformers import PreTrainedTokenizerFast
import numpy as np
import argparse
import os
from Bio import SeqIO

# Function to check if a sequence contains only valid DNA characters (A, T, C, G, N)
def validate_sequence(sequence):
    valid_characters = set("ATCGN")
    sequence_upper = sequence.upper()
    if not set(sequence_upper).issubset(valid_characters):
        raise ValueError(f"Invalid character(s) found in sequence: {sequence}")
    return sequence_upper

# Function to load the model
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = torch.load(model_path)
    model.config.output_hidden_states = True  # Enable output of hidden states
    model.eval()
    model = model.to(device)
    
    return model

# Function to load the tokenizer
def load_tokenizer(tokenizer_file):
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_file}' not found.")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    return tokenizer

# Function to calculate embeddings from the DNA sequence
def split_sequence(sequence, max_length):
    """Split a sequence into chunks of maximum length."""
    return [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]

def calculate_embeddings(model, tokenizer, sequence, device, max_seq_length=1024):
    """
    Calculate embeddings for a sequence by splitting if necessary.
    
    Args:
        model: The model to use for embedding calculation.
        tokenizer: The tokenizer to use for encoding the sequence.
        sequence (str): The input DNA or text sequence.
        device: The device to use for inference (e.g., 'cuda' or 'cpu').
        max_seq_length (int): The maximum sequence length supported by the model.
    
    Returns:
        np.ndarray: A 1D array containing the final embedding.
    """
    # Encode the entire sequence first
    encoded_sequence = tokenizer.encode(sequence.upper(), return_tensors='pt').to(device)
    print(f"Input IDs shape: {encoded_sequence.shape}")  # Debug statement
    
    if encoded_sequence.shape[1] > max_seq_length:
        print(f"Warning: Sequence too long ({encoded_sequence.shape[1]} tokens), splitting into chunks.")
        chunks = split_sequence(sequence, max_seq_length)
        embeddings = []
        
        for chunk in chunks:
            encoded_chunk = tokenizer.encode(chunk.upper(), return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model(encoded_chunk)
                # Ensure we get the last hidden state correctly
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state.cpu().numpy()
                elif hasattr(outputs, 'hidden_states') and isinstance(outputs.hidden_states, (list, tuple)):
                    hidden_states = outputs.hidden_states[-1].cpu().numpy()
                else:
                    raise ValueError("Could not find the last hidden state in the model output.")
                
                hidden_states_mean = np.mean(hidden_states, axis=1).reshape(-1)  # Compute mean along axis 1
                embeddings.append(hidden_states_mean)
        
        # Merge all chunk embeddings, e.g., by taking the mean
        final_embedding = np.mean(embeddings, axis=0)
    else:
        with torch.no_grad():
            outputs = model(encoded_sequence)
            # Ensure we get the last hidden state correctly
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state.cpu().numpy()
            elif hasattr(outputs, 'hidden_states') and isinstance(outputs.hidden_states, (list, tuple)):
                hidden_states = outputs.hidden_states[-1].cpu().numpy()
            else:
                raise ValueError("Could not find the last hidden state in the model output.")
            
            final_embedding = np.mean(hidden_states, axis=1).reshape(-1)  # Compute mean along axis 1
    
    return final_embedding

# Function to read sequences from a FASTA file
def read_fasta_sequences(fasta_file):
    sequences = []
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file '{fasta_file}' not found.")
    
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            validated_sequence = validate_sequence(str(record.seq))  # Validate the DNA sequence
            sequences.append(validated_sequence)
    
    return sequences

# Main function to handle input arguments and process the embeddings
def main():
    parser = argparse.ArgumentParser(description="Calculate model embeddings for DNA sequences in a FASTA file.")
    
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Path to the pretrained model and tokenizer file.")
    parser.add_argument("-f","--fasta_file", type=str, required=True, help="FASTA file containing DNA sequences.")
    parser.add_argument("-o","--output_file", type=str, default="embeddings.txt", help="Output file name for saving the embeddings.")
    
    args = parser.parse_args()

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU.")

    # Load model and tokenizer
    model_path = os.path.join(args.model_dir, 'pretrained_model.pt')
    tokenizer_path = os.path.join(args.model_dir, 'addgene_trained_dna_tokenizer.json')

    # Load model and tokenizer
    try:
        model = load_model(model_path, device)
        tokenizer = load_tokenizer(tokenizer_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Read sequences from the FASTA file
    try:
        sequences = read_fasta_sequences(args.fasta_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    embeddings = []

    # Calculate embeddings for each sequence
    for idx, sequence in enumerate(sequences):
        print(f"Processing sequence {idx+1}/{len(sequences)}")
        embedding = calculate_embeddings(model, tokenizer, sequence, device)
        embeddings.append(embedding)
    
    # Save the embeddings to a text file
    embeddings_np = np.array(embeddings)
    np.savetxt(args.output_file, embeddings_np, fmt='%.6f')
    print(f"Embeddings saved to {args.output_file}")

if __name__ == "__main__":
    main()
