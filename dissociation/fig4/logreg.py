import jax
import jax.numpy as jnp
from logistic import solve_softmax, predict_softmax
from logistic import crossval_softmax_predict

# Generate synthetic multiclass classification data
key = jax.random.PRNGKey(0)
n_samples = 1000
n_features = 50
n_classes = 3

# Create features
X = jax.random.normal(key, (n_samples, n_features))

# Create true coefficients for each class
true_beta = jax.random.normal(key, (n_features, n_classes))
true_intercept = jax.random.normal(key, (n_classes,))

# Generate multiclass probabilities using softmax
logits = jnp.dot(X, true_beta) + true_intercept
true_probs = jax.nn.softmax(logits, axis=1)

# Generate class labels
key, subkey = jax.random.split(key)
y = jax.random.categorical(subkey, logits)

# Convert y to one-hot encoding for softmax
y_onehot = jax.nn.one_hot(y, n_classes)

# Fit the model
coefficient, intercept = solve_softmax(X, y_onehot)

# Make predictions
y_pred = predict_softmax(X, y_onehot, X)

# Calculate accuracy
accuracy = jnp.mean(y_pred.argmax(-1) == y)

print(f"True coefficients shape: {true_beta.shape}")
print(f"True coefficients:\n{true_beta}")
print(f"\nFitted coefficients shape: {coefficient.shape}")
print(f"Fitted coefficients:\n{coefficient}")
print(f"\nTrue intercepts: {true_intercept}")
print(f"Fitted intercepts: {intercept.flatten()}")
print(f"\nAccuracy: {accuracy}")

# Test cross-validation predictions
cv_predictions = crossval_softmax_predict(X, y_onehot, num_splits=5, key=key)
cv_accuracy = jnp.mean(cv_predictions.argmax(-1) == y)

print("\nCross-validation results:")
print(f"CV predictions shape: {cv_predictions.shape}")
print(f"CV accuracy: {cv_accuracy}")
