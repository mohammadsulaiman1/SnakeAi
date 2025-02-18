import matplotlib.pyplot as plt
from IPython import display

# Enable interactive mode
plt.ion()

def plot(scores, mean_scores):
    """Plots and saves the training progress graph after each update."""
    
    # Clear previous plot and display new one
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Set plot title and labels
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot scores and mean scores
    plt.plot(scores, 'b-', label="Score per Game")  # Blue line for scores
    plt.plot(mean_scores, 'r-', label="Mean Score")  # Red line for mean scores
    
    # Ensure Y-axis starts at zero
    plt.ylim(ymin=0)
    
    # Add text annotations for latest score values
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]), fontsize=12, color='blue')
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), fontsize=12, color='red')

    # Show legend
    plt.legend()

    # Display updated plot
    plt.show(block=False)
    plt.pause(0.1)

    # ✅ Save the graph automatically after each update
    plt.savefig("training_progress.png")  # Saves as "training_progress.png"
