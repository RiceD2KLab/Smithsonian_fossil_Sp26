"""
Extracts and visualizes metadata from NDPI slides, particularly related to staining.
"""
import csv
import matplotlib.pyplot as plt

"""
Function for taking in csv metadata and producing a bar plot

Metadata: The first column should be the file name. The second column is an enum, "stained" or "no stained",
representing whether that file is a stained slide or not.

Output: Format and write to disk a matplotlib bar plot as a png

Example: plot_image_staining_distribution("Samples_with_annotations_stained_non_stained.csv")

"""

def plot_image_staining_distribution(file_name):
    files = []
    stained = []

    # Parse csv
    with open(file_name, 'r') as data:
        reader = csv.reader(data)
        next(reader)
        for row in reader:
            files.append(row[0])
            stained.append(row[1])

    # Reduce
    counts = [0,0]
    for s in stained:
        if s == "stained":
            counts[0] += 1
        else:
            counts[1] += 1

    # Produce bar plot
    fig, ax = plt.subplots()
    stained_labels = ['stained', 'not stained']
    bar_colors = ['tab:red', 'tab:blue']

    ax.bar(stained_labels, counts, color=bar_colors)

    ax.set_ylabel('occurrences')
    ax.set_title('Distribution of Image Staining')

    plt.savefig("image_metadata_bar_plot.png", format="png")
    plt.show()