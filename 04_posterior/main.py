import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Generate data for sine function with random noise e
def generate_data(n):
  x = np.random.uniform(size=(n,1))
  e = np.random.normal(0,0.08,x.shape)
  y = np.sin(2*np.pi*x)+e
  return (x,y.ravel())

x,y = generate_data(100)
fig, ax = plt.subplots(dpi=100)
plt.scatter(x,y)