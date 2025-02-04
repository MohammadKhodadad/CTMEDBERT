from utils.dataloader.get_negatives import save_hard_negatives, load_and_process_hard_negatives


directory = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs"  # Change this to your CSV directory
embedding_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/embeddings.npy"
output_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/hard_neg/hard_negative_v1.csv"

save_hard_negatives(directory, embedding_file=embedding_file)
load_and_process_hard_negatives(directory, embedding_file=embedding_file, output_file=output_file)