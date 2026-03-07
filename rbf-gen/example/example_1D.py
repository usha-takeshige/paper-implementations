"""Algorithm 1: RBF-Gen Training Procedure

This example follows Algorithm 1 from the paper step-by-step:

  "Knowledge-guided generative surrogate modeling for high-dimensional
   design optimization under scarce data"  (Wang et al.)

Problem
-------
Recover f(x) = 20x^2 + 20x + 1 on [-1, 1] from N=5 sparse observations.
Prior knowledge injected: the function is convex (d^2f/dx^2 >= 0).

Algorithm 1 Steps
-----------------
Step 1 : Construct RBF basis and interpolation system.
Step 2 : Compute null space (particular solution w0, null-space basis N).
Step 3 : Define generator G: z -> alpha that maps latent noise to null-space coefficients.
Step 4 : Incorporate prior knowledge (convexity penalty).
Step 5 : Train the generator by minimising the total loss.
"""

import torch
import matplotlib.pyplot as plt
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator
from rbf_gen.model import RBFGenModel
from rbf_gen.losses import RBFGenLoss, ConvexityPenalty
from rbf_gen.trainer import RBFGenTrainer

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dataset  D = {(x_i, y_i)}_{i=1}^N
# ---------------------------------------------------------------------------
def true_fn(x: torch.Tensor) -> torch.Tensor:
    return 20.0 * x**2 + 20.0 * x + 1.0


BOUNDS = torch.tensor([[-1.0], [1.0]])
N = 5
X_train = torch.linspace(-1.0, 1.0, N).unsqueeze(1)   # (N, 1)
y_train = true_fn(X_train.squeeze(1))                  # (N,)

print("=" * 60)
print("RBF-Gen  —  Algorithm 1 walkthrough")
print("=" * 60)
print(f"Training data: N={N}, domain=[-1, 1]")
print("True function: f(x) = 20x^2 + 20x + 1")
print()

# ===========================================================================
# Step 1: Construct RBF basis and interpolation system
#
#   Line 1: Place K RBF centers {c_j} uniformly / quasi-randomly in the design domain.
#   Line 2: Form the interpolation matrix  Phi_ij = phi(||x_i - c_j||).
# ===========================================================================
print("Step 1: Construct RBF basis and interpolation system")

K = 20                                  # K > N  (overcomplete)
kernel = GaussianKernel(epsilon=2.0)    # phi(r) = exp(-epsilon^2 * r^2)
rbf_basis = RBFBasis.from_quasi_random(K, BOUNDS, kernel)

Phi = rbf_basis.compute_matrix(X_train)  # (N, K)
print(f"  K={K} RBF centers placed (quasi-random Sobol)")
print(f"  Interpolation matrix Phi: {tuple(Phi.shape)}  (N x K)")
print()

# ===========================================================================
# Step 2: Compute null space of the interpolation system
#
#   Line 3: Solve  Phi w0 = y  for the particular (minimum-norm) solution w0.
#   Line 4: Compute null-space basis  N  such that  Phi N = 0,  N in R^{K x (K-N)}.
#   Line 5: Any admissible w = w0 + N alpha  satisfies Phi w = y automatically.
# ===========================================================================
print("Step 2: Compute null space")

null_dim = K - N   # dimension of the null space

# Compute once — injected into all subsequent models.
null_decomp = NullSpaceDecomposition()
null_decomp.fit(Phi, y_train)

w0         = null_decomp.w0          # (K,)
null_basis = null_decomp.null_basis  # (K, K-N)
print(f"  Particular solution w0: shape {tuple(w0.shape)}")
print(f"  Null-space basis N:     shape {tuple(null_basis.shape)}")
residual = (Phi @ w0 - y_train).abs().max().item()
print(f"  Max interpolation error of w0 on training data: {residual:.2e}")
print()

# ===========================================================================
# Step 3: Define generator for null-space exploration
#
#   Line 6: Initialise generator network  G(z; theta): z -> alpha,
#            where  z ~ N(0, I)  (latent_dim = null_dim = K - N).
#   Line 7: Each latent sample z yields a valid interpolant
#            f_z(x) = Phi(x)^T (w0 + N G(z)).
# ===========================================================================
print("Step 3: Define generator  G: z -> alpha")

# null_decomp is already computed in Step 2 — injected directly.
torch.manual_seed(42)
generator_trained = Generator(latent_dim=null_dim, null_dim=null_dim, hidden_dims=[32, 32])
model_trained = RBFGenModel(rbf_basis, null_decomp, generator_trained)

print(f"  Generator: z in R^{null_dim}  ->  alpha in R^{null_dim}")
print(f"  Architecture: {null_dim} -> 32 -> 32 -> {null_dim}  (Tanh activations)")
print()

# ===========================================================================
# Step 4: Incorporate prior knowledge
#
#   Line 8: Define loss terms for structural / distributional assumptions.
#   Line 9: Compose total loss  L = sum_k lambda_k * penalty_k + sum_l lambda_l * KL_l.
#
#   Domain knowledge used here: f is CONVEX (d^2f/dx^2 >= 0 everywhere).
# ===========================================================================
print("Step 4: Incorporate prior knowledge  (convexity)")

eval_grid = torch.linspace(-1.0, 1.0, 80).unsqueeze(1)   # evaluation grid for penalties

loss_fn = RBFGenLoss(
    penalty_terms=[
        ConvexityPenalty(convex=True, weight=1.0),  # penalise second_diff < 0
    ],
    kl_terms=[],  # no distributional constraints in this example
)
print("  Loss: L = lambda_convex * ConvexityPenalty")
print()

# ===========================================================================
# Step 5: Train the generator
#
#   Line 10: Optimise theta by training the generator neural network.
# ===========================================================================
print("Step 5: Train the generator")

trainer = RBFGenTrainer(
    model=model_trained,
    loss_fn=loss_fn,
    n_epochs=500,
    batch_size=32,
    eval_grid=eval_grid,
    lr=1e-3,
)
trainer.train()
print("  Training complete.")
print()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# Generate a model without constraints for comparison.
torch.manual_seed(42)
generator = Generator(
    latent_dim=null_dim,   # z dimension
    null_dim=null_dim,     # alpha = G(z) dimension  (= K - N)
    hidden_dims=[32, 32],  # small MLP
)
model_untrained = RBFGenModel(rbf_basis, null_decomp, generator)

print("Evaluation")
x_test = torch.linspace(-1.0, 1.0, 300).unsqueeze(1)
y_true = true_fn(x_test.squeeze(1))

N_SAMPLES = 500
with torch.no_grad():
    mean_unc = model_untrained.predict_mean(x_test, n_samples=N_SAMPLES)
    std_unc  = model_untrained.predict_std(x_test,  n_samples=N_SAMPLES)
    mean_con = model_trained.predict_mean(x_test,   n_samples=N_SAMPLES)
    std_con  = model_trained.predict_std(x_test,    n_samples=N_SAMPLES)

mse_unc = ((mean_unc - y_true) ** 2).mean().item()
mse_con = ((mean_con - y_true) ** 2).mean().item()
print(f"  Unconstrained  MSE : {mse_unc:.4f}  |  mean std : {std_unc.mean().item():.4f}")
print(f"  Constrained    MSE : {mse_con:.4f}  |  mean std : {std_con.mean().item():.4f}")
print()

# Interpolation check at training points
print("Interpolation check (error must be ~0 regardless of generator training):")
with torch.no_grad():
    y_at_train_unc = model_untrained.predict_mean(X_train, n_samples=N_SAMPLES)
    y_at_train_con = model_trained.predict_mean(X_train,   n_samples=N_SAMPLES)
for i, (xi, yi) in enumerate(zip(X_train.squeeze(1), y_train)):
    err_unc = (y_at_train_unc[i] - yi).abs().item()
    err_con = (y_at_train_con[i] - yi).abs().item()
    print(f"  x={xi:+.2f}  f_true={yi:7.3f}  "
          f"err_unc={err_unc:.2e}  err_con={err_con:.2e}")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
xs = x_test.squeeze(1).numpy()
yt = y_true.numpy()
xt = X_train.squeeze(1).numpy()
yt_train = y_train.numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

titles = [
    "Generator not trained\n(unconstrained null-space exploration)",
    "Generator trained with ConvexityPenalty\n(constrained null-space exploration)",
]
means = [mean_unc.numpy(), mean_con.numpy()]
stds  = [std_unc.numpy(),  std_con.numpy()]

for ax, title, ym, ys in zip(axes, titles, means, stds):
    ax.plot(xs, yt,  "k-",  linewidth=2.5, label="true f(x)", zorder=10)
    ax.plot(xs, ym,  "b--", linewidth=1.8, label="surrogate mean")
    ax.fill_between(xs, ym - 2 * ys, ym + 2 * ys,
                    alpha=0.25, color="blue", label=r"$\pm 2\sigma$")
    ax.scatter(xt, yt_train, color="red", zorder=11, s=80, label="training data")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("f(x)")

fig.suptitle(
    r"RBF-Gen  Algorithm 1  |  $f(x)=20x^2+20x+1$,  "
    rf"N={N} pts,  K={K} centers",
    fontsize=13,
)
plt.tight_layout()

out_path = "example/example_1D.png"
plt.savefig(out_path, dpi=130)
print(f"\nPlot saved to {out_path}")
