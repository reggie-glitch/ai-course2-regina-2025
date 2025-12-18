import matplotlib.pyplot as plt
import seaborn as sns

def create_histogram(df, column, output_path):
    """Create histogram."""
    plt.figure(figsize=(10, 6))
    df[column].hist()
    plt.title(f'Histogram of {column}')
    plt.savefig(output_path)
    plt.close()
