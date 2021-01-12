"""
Labels for Soft Ordinal Regression (SORD)
"""

import numpy as np


def soft_label_theta(theta_true, n_bins=100, K=1):
    theta_pred = np.linspace(-np.pi/2, np.pi/2, n_bins, endpoint=False)
    th = np.array(theta_true).reshape(-1,1)    
    m1 = np.abs(theta_pred-th)
    m2 = np.abs(np.fmod(theta_pred+1*np.pi, 2*np.pi) - th)
    m3 = np.abs(np.fmod(theta_pred-1*np.pi, 2*np.pi) - th)
    phi = K * np.minimum.reduce([m1,m2,m3])
    labels = np.exp(-(phi**2), dtype=np.float32)
    return labels / labels.sum(axis=1,keepdims=True)


def soft_label_rho(rho_true, n_bins=100, K=1, K_range=1):
    rho_pred = np.tan(np.linspace(-np.pi/2, np.pi/2, n_bins, endpoint=True)) * K_range
    rho = np.array(rho_true).reshape(-1,1)
    d = (rho - rho_pred) * K
    labels = np.exp(-(d ** 2))
    return labels / labels.sum(axis=1,keepdims=True)


def theta_from_soft_label(theta_pred):
    n_bins = theta_pred.shape[1]
    theta = np.linspace(-np.pi/2, np.pi/2, n_bins, endpoint=False)
    t = np.where(theta_pred<0.01, 0, theta_pred)
    x = (t * np.cos(theta).reshape(1,-1)).sum(-1)
    y = (t * np.sin(theta).reshape(1,-1)).sum(-1)
    d = np.array([x,y]).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    theta = np.arccos(d @ [0,1])
    return theta


def rho_from_soft_label(rho_pred, K_range=1):
    n_bins = rho_pred.shape[1]
    rho_bins = np.tan(np.linspace(-np.pi/2, np.pi/2, n_bins, endpoint=True)) * K_range
    rho_bins = rho_bins.reshape(1,-1)
    r = np.where(rho_pred<0.1*rho_pred.max(-1,keepdims=True), 0, rho_pred)
    r /= r.sum(axis=-1, keepdims=True)
    return (r * rho_bins).sum(-1)