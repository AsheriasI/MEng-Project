# %% [markdown]
# # Standalone Notebook: Mutual Information Analysis for the OAK Model
#
# This notebook:
#
# 1. Generates two synthetic datasets using functions f1 and f2.
# 2. Splits one dataset into training and testing sets.
# 3. Trains an Orthogonal Additive Kernel (OAK) model.
# 4. Computes predictions, Sobol indices, and per-component outputs.
# 5. Calculates and plots the mutual information (MI) for each kernel component.
#
# The MI calculations use the following Gaussian formula:
#
# \[
# MI_i = \frac{1}{2}\ln\!\left(\frac{V_{\text{total}}}{V_{\text{total}} - \sigma_i^2}\right)
# \]
#
# where \(\sigma_i^2\) is the variance explained by kernel component \(i\) and
# \(V_{\text{total}}\) is the total variance of the model predictions.
#
# *References:* Cover & Thomas (Elements of Information Theory), Rasmussen & Williams (Gaussian Processes for Machine Learning).

# %% 
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path

# Import necessary OAK modules (assuming the oak package is installed)
from oak.model_utils import oak_model, save_model
from oak.utils import (
    get_model_sufficient_statistics,
    get_prediction_component,
    extract_active_dims,
    get_list_representation,
    model_to_kernel_list,
)

# Set plotting preferences and random seeds for reproducibility
matplotlib.rcParams.update({"font.size": 20})
np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})
np.random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## Define Synthetic Functions
#
# We define two functions:
#
# - **f1:** A smooth additive function:  
#   \(\sin(2\pi x_1) + x_2^2 + 0.5x_3\)
#
# - **f2:** A function with a localized bump:  
#   \(f1(x) + \exp(-30 \|x\|^2)\)

# %%
def f1(X):
    """Smooth function: sin(2π*x₁) + x₂² + 0.5x₃"""
    return np.sin(2 * np.pi * X[:, 0]) + X[:, 1]**2 + 0.5 * X[:, 2]

def f2(X):
    """Less smooth function with a localized bump"""
    base = np.sin(2 * np.pi * X[:, 0]) + X[:, 1]**2 + 0.5 * X[:, 2]
    bump = np.exp(-30 * np.sum(X**2, axis=1))
    return base + bump

# %% [markdown]
# ## Generate Synthetic Datasets
#
# We generate two datasets with \(N = 500\) samples and \(D = 3\) features,
# sampling uniformly from [0, 1]^3. Dataset f1 uses the smooth function, and
# dataset f2 uses the function with the bump.
#
# For this notebook we will work with the f2 dataset.

# %%
N = 500  # number of samples
D = 3    # number of features
X = np.random.uniform(0, 1, (N, D))

y1 = f1(X).reshape(-1, 1)  # dataset using f1
y2 = f2(X).reshape(-1, 1)  # dataset using f2

# Select one dataset to work with; here we use f2.
X_data, y_data = X, y2

# %% [markdown]
# ## Split Data into Training and Testing Sets
#
# We use 5-fold cross-validation and select the first fold for this example.

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for train_index, test_index in kf.split(X_data):
    if fold == 0:
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
    fold += 1

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# %% [markdown]
# ## Train the Orthogonal Additive Kernel (OAK) Model
#
# We create an instance of the OAK model with maximum interaction depth equal to
# the number of features, and then fit it on the training data.

# %%
oak = oak_model(max_interaction_depth=X_train.shape[1])
oak.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Model on Test Data
#
# We clip test inputs to the range of the training data (to avoid extrapolation),
# then predict using the OAK model and compute performance metrics.

# %%
x_min, x_max = X_train.min(0), X_train.max(0)
X_test_clipped = np.clip(X_test, x_min, x_max)

# Predict using the OAK model
y_pred = oak.predict(X_test_clipped)

# Compute metrics: RSS, TSS, R2, RMSE
rss = ((y_pred - y_test[:, 0]) ** 2).mean()
tss = ((y_test[:, 0] - y_test[:, 0].mean()) ** 2).mean()
r2 = 1 - rss / tss
rmse = np.sqrt(rss)
print(f"R2 = {r2:.5f}")
print(f"RMSE = {rmse:.5f}")

# %% [markdown]
# ## Calculate Sobol Indices for Each Kernel Component
#
# We then calculate the Sobol indices using the OAK model’s method.
# These indices provide the normalized variance contribution of each kernel component.

# %%
print("Calculating Sobol indices:")
oak.get_sobol()
tuple_of_indices = oak.tuple_of_indices
normalised_sobols = oak.normalised_sobols
print(f"Normalized Sobol indices: {normalised_sobols}")
print(f"Kernel component indices: {tuple_of_indices}")

# %% [markdown]
# ## Get Predictions for Each Kernel Component
#
# We extract the predictions corresponding to each kernel component.
# These contributions are later used in the mutual information calculations.

# %%
# Transform test inputs and get sufficient statistics for prediction
XT = oak._transform_x(X_test_clipped)
oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)

# Get predicted contributions for each kernel component
prediction_list = get_prediction_component(oak.m, oak.alpha, XT)

# Compute the constant term (bias) from the model.
constant_term = oak.alpha.numpy().sum() * oak.m.kernel.variances[0].numpy()
print(f"Constant term: {constant_term:.5f}")
y_pred_component = np.ones(y_test.shape[0]) * constant_term

# Cumulatively add the predictions of each component (ordered by normalized Sobol indices)
cumulative_sobol, rmse_component = [], []
order = np.argsort(normalised_sobols)[::-1]
for n in order:
    y_pred_component += prediction_list[n].numpy()
    y_pred_component_transformed = oak.scaler_y.inverse_transform(y_pred_component.reshape(-1, 1))
    error_component = np.sqrt(((y_pred_component_transformed - y_test) ** 2).mean())
    rmse_component.append(error_component)
    cumulative_sobol.append(normalised_sobols[n])
cumulative_sobol = np.cumsum(cumulative_sobol)

# Sanity check: cumulative prediction should match overall model prediction
np.testing.assert_allclose(y_pred_component_transformed[:, 0], y_pred)

# %% [markdown]
# ## Mutual Information Calculations for OAK Model Components
#
# We now calculate the mutual information (MI) for each kernel component.
#
# Let:
# - \(V_{\text{total}} =\) total variance of model predictions.
# - \(\sigma_i^2 = (\text{normalized Sobol index for component } i) \times V_{\text{total}}\).
#
# Then the MI for component \(i\) is:
#
# \[
# MI_i = \frac{1}{2}\ln\!\left(\frac{V_{\text{total}}}{V_{\text{total}} - \sigma_i^2}\right)
# \]
#
# This is computed in nats and then converted to bits.

# %%
# Total variance from the model's predictions
V_total = np.var(y_pred)
print(f"Total Variance (V_total): {V_total:.5f}")

# Compute the variance explained by each component
sigma_components = np.array(normalised_sobols) * V_total

# Calculate MI for each component (in nats)
MI_components_nats = 0.5 * np.log( V_total / (V_total - sigma_components) )
print("Mutual Information for each kernel component (in nats):")
for i, mi in enumerate(MI_components_nats):
    print(f"Component {tuple_of_indices[i]}: MI = {mi:.5f} nats")

# Convert MI from nats to bits (1 nat = 1/ln(2) bits)
MI_components_bits = MI_components_nats / np.log(2)
print("\nMutual Information for each kernel component (in bits):")
for i, mi in enumerate(MI_components_bits):
    print(f"Component {tuple_of_indices[i]}: MI = {mi:.5f} bits")

# %% [markdown]
# ## Plot Mutual Information per Kernel Component

# %%
component_labels = [str(tup) for tup in tuple_of_indices]
plt.figure(figsize=(10, 6))
plt.bar(component_labels, MI_components_bits, color='skyblue', edgecolor='black')
plt.xlabel("Kernel Component (Active Dimensions)")
plt.ylabel("Mutual Information (bits)")
plt.title("Mutual Information of OAK Model Components")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Joint Mutual Information Calculation
#
# For example, we compute the joint MI for all components that include \(x_1\) (i.e. dimension 0).
#
# For a set \(S\) of components, the joint MI is:
#
# \[
# MI_S = \frac{1}{2}\ln\!\left(\frac{V_{\text{total}}}{V_{\text{total}} - \sum_{i\in S} \sigma_i^2}\right)
# \]

# %%
selected_indices = [i for i, dims in enumerate(tuple_of_indices) if 0 in dims]
if len(selected_indices) > 0:
    sigma_joint = np.sum(sigma_components[selected_indices])
    MI_joint_nats = 0.5 * np.log( V_total / (V_total - sigma_joint) )
    MI_joint_bits = MI_joint_nats / np.log(2)
    print(f"Joint Mutual Information for components including x1: {MI_joint_nats:.5f} nats, or {MI_joint_bits:.5f} bits")
else:
    print("No components include x1.")

# %% [markdown]
# # Summary
#
# In this notebook, we:
#
# 1. Generated two synthetic datasets using f1 and f2.
# 2. Split one dataset (using f2) into training and testing sets.
# 3. Trained an Orthogonal Additive Kernel (OAK) model.
# 4. Computed predictions and extracted Sobol indices and kernel component contributions.
# 5. Calculated and visualized the mutual information for each kernel component, and computed joint MI for components including \(x_1\).
#
# This notebook provides a foundation for further simulation and analysis of mutual information for GP kernels.
