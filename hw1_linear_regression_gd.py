#!/usr/bin/env python3
"""
HW1: Linear Regression with Gradient Descent (from scratch)
Author: <Your Name>
Course: ECGR3123 / Data Communications and Networking (adjust if needed)
File: hw1_linear_regression_gd.py

What this script does
---------------------
1) Loads a dataset D3.csv with 4 columns: x1, x2, x3, y.
2) Problem 1 (Univariate): Trains 3 separate linear models y = b0 + b1 * xi
   using gradient descent (no ML libraries) for i in {1,2,3}. Tries several
   learning rates; picks the best by final loss. Produces:
   - Final model (b0, b1) for each xi
   - A scatter+line plot for each xi
   - A loss-vs-iteration plot for each learning rate tried
   - Summary table of losses, iterations, and whether runs diverged
3) Problem 2 (Multivariate): Trains y = b0 + b1*x1 + b2*x2 + b3*x3
   using gradient descent across several learning rates. Produces:
   - Final model (b0..b3)
   - Loss plot
   - Summary table of lr vs convergence
   - Predictions for x_new = (1,1,1), (2,0,4), (3,2,1)
4) Saves all figures and a single PDF "hw1_results.pdf" that includes:
   - A text cover page (what was done)
   - All plots and summaries
   - A code listing at the end of the PDF so your "code+plots" PDF requirement is met

How to run
----------
$ python hw1_linear_regression_gd.py --csv D3.csv
Optional args:
  --alphas 0.1 0.05 0.02 0.01   # learning rates to try (default)
  --max-iters 5000              # max iterations per run
  --tol 1e-8                    # stopping tolerance on relative loss change
  --no-standardize              # turn OFF feature standardization (default is ON)
  --outdir results              # output folder (created if missing)

NOTE: Feature standardization is ON by default for numerical stability.
If you must match a "raw features" interpretation, pass --no-standardize.
Predictions are always reported on the original (unscaled) feature space.
"""

import argparse
import numpy as np
import pandas as pd
import math
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 4:
        raise ValueError("Expected 4 columns: x1, x2, x3, y")
    X = df.iloc[:, 0:3].to_numpy(dtype=float)
    y = df.iloc[:, 3].to_numpy(dtype=float)
    return X, y

def add_intercept(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    ones = np.ones((m, 1), dtype=float)
    return np.hstack([ones, X])

def mse_loss(Xd: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    preds = Xd @ theta
    err = preds - y
    return (err @ err) / (2.0 * m)

def gradient_descent(Xd: np.ndarray, y: np.ndarray, alpha: float, max_iters: int = 5000, tol: float = 1e-8,
                     theta0: np.ndarray = None, early_stop_window: int = 10) -> Dict:
    m, n = Xd.shape
    theta = np.zeros(n) if theta0 is None else theta0.copy()
    history = []
    last_loss = None
    diverged = False

    for t in range(1, max_iters + 1):
        preds = Xd @ theta
        gradient = (Xd.T @ (preds - y)) / m
        theta -= alpha * gradient
        loss = mse_loss(Xd, y, theta)
        history.append(loss)

        # Detect divergence (loss becomes nan/inf or grows explosively)
        if not np.isfinite(loss) or (last_loss is not None and loss > last_loss * 1.5):
            diverged = True
            break

        # Relative improvement stopping
        if last_loss is not None:
            rel = abs(last_loss - loss) / max(1.0, last_loss)
            if rel < tol and t > early_stop_window:
                break

        last_loss = loss

    return {"theta": theta, "history": history, "iters": len(history), "diverged": diverged}

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma

def unstandardize_theta(theta_scaled: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    For model y = b0 + sum(bj * xj) learned on standardized Xs = (X-mu)/sigma,
    recover equivalent coefficients on original X:
      b0_orig = b0_scaled - sum(bj_scaled * mu_j / sigma_j)
      bj_orig = bj_scaled / sigma_j
    """
    b0 = theta_scaled[0]
    b = theta_scaled[1:]
    b_orig = b / sigma
    b0_orig = b0 - np.sum(b * (mu / sigma))
    return np.concatenate([[b0_orig], b_orig])

def run_univariate(X: np.ndarray, y: np.ndarray, alphas: List[float], max_iters: int, tol: float,
                   standardize: bool, outdir: str, pdf: PdfPages) -> Dict[str, Dict]:
    results = {}
    m = X.shape[0]

    for j in range(3):
        xj = X[:, [j]]
        label = f"X{j+1}"

        if standardize:
            xs, mu, sigma = standardize_fit(xj)
            Xd = add_intercept(xs)
        else:
            mu, sigma = np.array([0.0]), np.array([1.0])
            Xd = add_intercept(xj)

        # Try multiple learning rates
        tries = []
        for a in alphas:
            out = gradient_descent(Xd, y, alpha=a, max_iters=max_iters, tol=tol)
            out["alpha"] = a
            tries.append(out)

        # Pick best by final loss (among non-diverged), else keep the least bad
        valid = [t for t in tries if not t["diverged"]]
        chosen = min(valid, key=lambda t: t["history"][-1]) if valid else min(tries, key=lambda t: t["history"][-1])
        theta_scaled = chosen["theta"]
        if standardize:
            theta_orig = unstandardize_theta(theta_scaled, mu.flatten(), sigma.flatten())
        else:
            theta_orig = theta_scaled

        results[label] = {
            "theta": theta_orig,
            "chosen": chosen,
            "all_runs": tries,
            "mu": mu.flatten(),
            "sigma": sigma.flatten(),
        }

        # --- Plots: data + fitted line (original scale) ---
        # Build line across the min..max of xj original scale
        x_min, x_max = xj.min(), xj.max()
        xs_line = np.linspace(x_min, x_max, 200)
        ys_line = theta_orig[0] + theta_orig[1] * xs_line

        fig = plt.figure()
        plt.scatter(xj, y, s=16)
        plt.plot(xs_line, ys_line, linewidth=2)
        plt.xlabel(label)
        plt.ylabel("Y")
        plt.title(f"Problem 1: Best Fit using {label} (alpha={chosen['alpha']}, iters={chosen['iters']})")
        fig_path = os.path.join(outdir, f"p1_fit_{label.lower()}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Loss curves for each learning rate (one figure per lr) ---
        for t in tries:
            fig2 = plt.figure()
            plt.plot(range(1, len(t["history"]) + 1), t["history"], linewidth=1.5)
            status = "diverged" if t["diverged"] else "ok"
            plt.xlabel("Iteration")
            plt.ylabel("Loss (MSE/2)")
            plt.title(f"Problem 1: {label} Loss Curve (alpha={t['alpha']}, {status}, iters={t['iters']})")
            fig2_path = os.path.join(outdir, f"p1_loss_{label.lower()}_a{str(t['alpha']).replace('.','p')}.png")
            plt.savefig(fig2_path, bbox_inches="tight")
            pdf.savefig(fig2)
            plt.close(fig2)

        # --- Summary page as text ---
        lines = [f"Problem 1 – {label} Summary",
                 f"Best alpha: {chosen['alpha']} | iterations: {chosen['iters']} | final loss: {chosen['history'][-1]:.6g}",
                 f"Final model (original scale): y = {theta_orig[0]:.6g} + ({theta_orig[1]:.6g})*{label}",
                 "",
                 "All trials (alpha, iters, final loss, diverged):"]
        for t in tries:
            lines.append(f"  {t['alpha']:>8g} | {t['iters']:>6d} | {t['history'][-1]:.6g} | {'YES' if t['diverged'] else 'NO'}")

        fig3 = plt.figure()
        plt.axis("off")
        plt.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")
        pdf.savefig(fig3); plt.close(fig3)

    # Page ranking variables by loss
    ranking = sorted([(lbl, res["chosen"]["history"][-1]) for lbl, res in results.items()], key=lambda x: x[1])
    rank_lines = ["Problem 1 – Which variable explains Y best (lowest final loss)?"]
    for r, (lbl, loss) in enumerate(ranking, start=1):
        rank_lines.append(f"{r}. {lbl} with loss {loss:.6g}")

    fig4 = plt.figure()
    plt.axis("off")
    plt.text(0.01, 0.98, "\n".join(rank_lines), va="top", family="monospace")
    pdf.savefig(fig4); plt.close(fig4)

    return results

def run_multivariate(X: np.ndarray, y: np.ndarray, alphas: List[float], max_iters: int, tol: float,
                     standardize: bool, outdir: str, pdf: PdfPages) -> Dict:
    if standardize:
        Xs, mu, sigma = standardize_fit(X)
        Xd = add_intercept(Xs)
    else:
        mu = np.zeros(X.shape[1])
        sigma = np.ones(X.shape[1])
        Xd = add_intercept(X)

    tries = []
    for a in alphas:
        out = gradient_descent(Xd, y, alpha=a, max_iters=max_iters, tol=tol)
        out["alpha"] = a
        tries.append(out)

    valid = [t for t in tries if not t["diverged"]]
    chosen = min(valid, key=lambda t: t["history"][-1]) if valid else min(tries, key=lambda t: t["history"][-1])

    theta_scaled = chosen["theta"]
    if standardize:
        theta_orig = unstandardize_theta(theta_scaled, mu, sigma)
    else:
        theta_orig = theta_scaled

    # Loss curves per alpha
    for t in tries:
        fig = plt.figure()
        plt.plot(range(1, len(t["history"]) + 1), t["history"], linewidth=1.5)
        status = "diverged" if t["diverged"] else "ok"
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE/2)")
        plt.title(f"Problem 2: Multivariate Loss (alpha={t['alpha']}, {status}, iters={t['iters']})")
        plt.savefig(os.path.join(outdir, f"p2_loss_a{str(t['alpha']).replace('.','p')}.png"), bbox_inches="tight")
        pdf.savefig(fig); plt.close(fig)

    # Summary with coefficients
    lines = [
        "Problem 2 – Multivariate Summary",
        f"Best alpha: {chosen['alpha']} | iterations: {chosen['iters']} | final loss: {chosen['history'][-1]:.6g}",
        "Final model (original scale):",
        f"y = {theta_orig[0]:.6g} + ({theta_orig[1]:.6g})*X1 + ({theta_orig[2]:.6g})*X2 + ({theta_orig[3]:.6g})*X3",
        "",
        "All trials (alpha, iters, final loss, diverged):"
    ]
    for t in tries:
        lines.append(f"  {t['alpha']:>8g} | {t['iters']:>6d} | {t['history'][-1]:.6g} | {'YES' if t['diverged'] else 'NO'}")

    fig2 = plt.figure()
    plt.axis("off")
    plt.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")
    pdf.savefig(fig2); plt.close(fig2)

    return {"theta": theta_orig, "chosen": chosen, "all_runs": tries}

def add_cover_page(pdf: PdfPages, course: str, student: str, notes: str):
    fig = plt.figure()
    plt.axis("off")
    txt = f"""Homework 1 – Linear Regression with Gradient Descent
{course}
Student: {student}

This PDF includes:
• Problem 1: three univariate models (X1, X2, X3) with data+fit plots and loss curves.
• A ranking of which single variable best explains Y (lowest loss).
• Problem 2: multivariate model with loss curves.
• Final coefficients and predictions for new inputs.
• A code listing for this script at the end.

Notes:
{notes}
"""
    plt.text(0.05, 0.95, txt, va="top")
    pdf.savefig(fig); plt.close(fig)

def add_predictions_page(pdf: PdfPages, theta: np.ndarray, X_new: List[Tuple[float, float, float]]):
    lines = ["Problem 2 – Predictions",
             f"Model: y = {theta[0]:.6g} + ({theta[1]:.6g})*X1 + ({theta[2]:.6g})*X2 + ({theta[3]:.6g})*X3",
             ""]
    for vec in X_new:
        x1, x2, x3 = vec
        yhat = theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x3
        lines.append(f"Input (X1,X2,X3) = {vec} -> predicted y = {yhat:.6g}")

    fig = plt.figure()
    plt.axis("off")
    plt.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")
    pdf.savefig(fig); plt.close(fig)

def add_code_listing(pdf: PdfPages, path: str):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read().splitlines()

    # chunk the code into pages
    chunk = 58
    for i in range(0, len(code), chunk):
        fig = plt.figure()
        plt.axis("off")
        page = "\n".join(code[i:i+chunk])
        plt.text(0.01, 0.98, page, va="top", family="monospace")
        plt.title(f"Code listing ({os.path.basename(path)}) page {i//chunk + 1}")
        pdf.savefig(fig); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="D3.csv", help="Path to D3.csv (4 cols: x1,x2,x3,y)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05, 0.02, 0.01], help="Learning rates to try")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--course", type=str, default="(Course name here)")
    parser.add_argument("--student", type=str, default="(Your name here)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X, y = load_dataset(args.csv)
    standardize = not args.no_standardize

    pdf_path = os.path.join(args.outdir, "hw1_results.pdf")
    with PdfPages(pdf_path) as pdf:
        add_cover_page(pdf, args.course, args.student,
                       notes=f"CSV: {args.csv}\nAlphas tried: {args.alphas}\nStandardization: {'ON' if standardize else 'OFF'}")

        # Problem 1
        p1 = run_univariate(X, y, args.alphas, args.max_iters, args.tol, standardize, args.outdir, pdf)

        # Problem 2
        p2 = run_multivariate(X, y, args.alphas, args.max_iters, args.tol, standardize, args.outdir, pdf)

        # Predictions for specified new points
        X_new = [(1,1,1), (2,0,4), (3,2,1)]
        add_predictions_page(pdf, p2["theta"], X_new)

        # Code listing
        add_code_listing(pdf, __file__)

    # Also write a text summary for your report convenience
    best_uni = min(((k, v["chosen"]["history"][-1]) for k, v in p1.items()), key=lambda x: x[1])
    summary_lines = [
        "HW1 Summary",
        f"Best univariate variable: {best_uni[0]} (loss={best_uni[1]:.6g})",
        f"Multivariate best alpha: {p2['chosen']['alpha']} | iterations: {p2['chosen']['iters']} | final loss: {p2['chosen']['history'][-1]:.6g}",
        f"Final multivariate model: y = {p2['theta'][0]:.6g} + ({p2['theta'][1]:.6g})*X1 + ({p2['theta'][2]:.6g})*X2 + ({p2['theta'][3]:.6g})*X3",
    ]
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Done!")
    print(f"Output folder: {args.outdir}")
    print(f"Main PDF with plots+code: {pdf_path}")
    print("Text summary saved to summary.txt")
    print("Tip: include the PDF in your submission and link to your GitHub repo containing this script.")

if __name__ == "__main__":
    main()
