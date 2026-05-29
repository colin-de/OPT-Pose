# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import numpy as np


def umeyama_similarity_transform(X, Y, point_conf=None, ransac=False, ransac_iters=200, inlier_thresh=0.02, fix_scale=False):
    """
    Weighted Umeyama / Procrustes estimator with optional RANSAC.

    Args:
        X: (N,3) source points
        Y: (N,3) target points
        point_conf: (N,) optional confidences/weights for points. If provided, used as weights and sampling probs for RANSAC.
        ransac: bool, whether to run RANSAC to robustly estimate the transform.
        ransac_iters: number of RANSAC iterations.
        inlier_thresh: distance threshold (meters) to consider an inlier.
        fix_scale: if True, enforce s=1 (pure SE3). If False, estimate scale (Sim3).

    Returns:
        R: (3,3) rotation
        t: (3,) translation
        s: scalar scale (1.0 if fix_scale=True)
        T: (4,4) homogeneous transform (T[:3,:3] = s*R)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape == Y.shape, "Point sets must have same shape"

    N = X.shape[0]
    if N == 0:
        # empty, return identity
        R = np.eye(3)
        t = np.zeros(3)
        s = 1.0
        T = np.eye(4)
        return R, t, s, T

    if point_conf is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        pc = np.asarray(point_conf, dtype=np.float64)
        if pc.shape[0] != N:
            # try to flatten
            pc = pc.flatten()[:N]
        weights = pc.copy()
        weights[weights < 0] = 0.0

    # Prevent all-zero weights
    if weights.sum() <= 0:
        weights = np.ones(N, dtype=np.float64)

    def compute_transform_from_inliers(idx, use_weights=True):
        Xs = X[idx]
        Ys = Y[idx]
        ws = weights[idx] if use_weights else np.ones(len(idx), dtype=np.float64)
        W = ws.sum()
        # weighted centroids
        mean_X = (ws[:, None] * Xs).sum(axis=0) / W
        mean_Y = (ws[:, None] * Ys).sum(axis=0) / W

        Xc = Xs - mean_X
        Yc = Ys - mean_Y

        H = Xc.T @ (ws[:, None] * Yc)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        if fix_scale:
            s = 1.0
        else:
            denom = (ws * np.sum(Xc ** 2, axis=1)).sum()
            if denom > 1e-12:
                s = S.sum() / denom
            else:
                s = 1.0

        t = mean_Y - s * (R @ mean_X)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = s * R
        T[:3, 3] = t
        return R, t, s, T

    if ransac and N >= 3:
        best_inliers = None
        best_inlier_count = 0
        best_model = None

        # sampling probabilities from weights
        probs = weights / weights.sum()

        for _ in range(ransac_iters):
            try:
                sample_k = min(3, N)
                sample_idx = np.random.choice(N, size=sample_k, replace=False, p=probs)
            except Exception:
                sample_idx = np.random.choice(N, size=min(3, N), replace=False)

            R_s, t_s, s_s, T_s = compute_transform_from_inliers(sample_idx, use_weights=False)

            # compute residuals for all points
            X_trans = (s_s * (R_s @ X.T).T) + t_s
            residuals = np.linalg.norm(X_trans - Y, axis=1)
            inliers = residuals < inlier_thresh
            inlier_count = int(inliers.sum())

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_model = (R_s, t_s, s_s, T_s)

                # early exit if all points are inliers
                if inlier_count == N:
                    break

        if best_inliers is None:
            # fallback to full fit
            R, t, s, T = compute_transform_from_inliers(np.arange(N), use_weights=True)
            return R, t, s, T

        # refine using all inliers and use weights
        inlier_idx = np.nonzero(best_inliers)[0]
        R, t, s, T = compute_transform_from_inliers(inlier_idx, use_weights=True)
        return R, t, s, T
    else:
        # single closed-form weighted solution
        R, t, s, T = compute_transform_from_inliers(np.arange(N), use_weights=True)
        return R, t, s, T