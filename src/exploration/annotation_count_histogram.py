"""
Plots a histogram of the number of annotated palynomorphs per image. 
"""
import matplotlib.pyplot as plt

with open("annotation_counts.txt", 'r') as file:
    data = [int(line.split()[-1]) for line in file]

plt.hist(data, bins='auto')

plt.xlabel("Number of Annotated Palynomorphs")
plt.ylabel("Number of Images")
plt.title("Distribution of Annotations per Image")

plt.savefig("annotation_count_histogram.png")
plt.show()