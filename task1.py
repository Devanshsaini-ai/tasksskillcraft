# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic age data
np.random.seed(42)
ages = np.random.normal(loc=35, scale=10, size=200)
ages = np.clip(ages, 0, 90)  

plt.figure(figsize=(10, 6))
plt.hist(ages, bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution in a Population')
plt.xlabel('Age')
plt.ylabel('Number of Individuals')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

