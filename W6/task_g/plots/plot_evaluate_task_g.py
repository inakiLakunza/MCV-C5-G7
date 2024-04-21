import matplotlib.pyplot as plt

# USED OUTPUT OF evaluate.py USING predictions_multimodal_test_set.csv


# Age distribution
ages = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61+']
age_counts = [5, 48, 410, 972, 458, 64, 18]

# Average accuracy of each age category
age_categories = ['1', '2', '3', '4', '5', '6', '7']
avg_accuracy = [0.0000, 0.0000, 0.0610, 0.7593, 0.1528, 0.0000, 0.0000]

# Plotting age distribution
plt.figure(figsize=(10, 5))
plt.bar(ages, age_counts, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig("age_dist_task_g.png")
plt.clf()

# Plotting average accuracy of each age category
plt.figure(figsize=(10, 5))
plt.bar(age_categories, avg_accuracy, color='lightgreen')
plt.title('Average Accuracy by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)  # Limit y-axis to range [0, 1]
plt.savefig("avg_accuracy_task_g.png")