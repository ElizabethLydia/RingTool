import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error (MAE)"""
    abs_errors = torch.abs(predictions - targets)
    mae = torch.mean(abs_errors).item()
    standard_error = torch.std(abs_errors).item() / torch.sqrt(torch.tensor(predictions.numel()))
    return mae, standard_error

def calculate_rmse(predictions, targets):
    """Calculate Root Mean Square Error (RMSE)"""
    squared_errors = (predictions - targets) ** 2
    rmse = torch.sqrt(torch.mean(squared_errors)).item()
    standard_error = torch.sqrt(torch.std(squared_errors) / torch.sqrt(torch.tensor(predictions.numel()))).item()
    return rmse, standard_error

def calculate_mape(predictions, targets):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    mask = targets != 0
    if mask.sum() > 0:
        percent_errors = torch.abs((targets[mask] - predictions[mask]) / targets[mask])
        mape = torch.mean(percent_errors) * 100
        standard_error = torch.std(percent_errors) / torch.sqrt(torch.tensor(mask.sum())) * 100
        return mape.item(), standard_error.item()
    return float('inf'), float('inf')

'''
def calculate_pearson(predictions, targets):
    """Calculate Pearson correlation coefficient"""
    x = predictions.flatten()
    y = targets.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # Calculate standard error using the formula: sqrt((1-r²)/(n-2))
    n = torch.tensor(x.numel())
    if n > 2:
        standard_error = torch.sqrt((1 - pearson**2) / (n - 2))
        return pearson.item(), standard_error.item()
    return pearson.item(), float('inf')
'''
    
def calculate_pearson(x, y, eps=1e-5):
    # Flatten tensors to ensure we're computing a single correlation value
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate means
    x_mean = torch.mean(x_flat)
    y_mean = torch.mean(y_flat)
    
    # Calculate Pearson correlation
    numerator = torch.sum((x_flat - x_mean) * (y_flat - y_mean))
    denominator = torch.sqrt(torch.sum((x_flat - x_mean) ** 2) * torch.sum((y_flat - y_mean) ** 2) + eps)
    
    pearson = numerator / denominator
    
    # Calculate standard error
    n = torch.tensor(x_flat.numel())
    if n > 2:
        standard_error = torch.sqrt((1 - pearson**2) / (n - 2))
        return pearson.item(), standard_error.item()
    return pearson.item(), float('inf')

def value_with_std(value, std):
    """Return value with standard error: value±std"""
    return f"{value:.2f}±{std:.2f}"

def calculate_metrics(predictions, targets):
    """Calculate all metrics"""
    mae, mae_std = calculate_mae(predictions, targets)
    rmse, rmse_std = calculate_rmse(predictions, targets)
    mape, mape_std = calculate_mape(predictions, targets)
    pearson, pearson_std = calculate_pearson(predictions, targets)
    sample_len = len(predictions)
    return {
        "sample_len": sample_len,
        "mae": mae,
        "mae_with_std": value_with_std(mae, mae_std),
        "rmse": rmse,
        "rmse_with_std": value_with_std(rmse, rmse_std),
        "mape": mape,
        "mape_with_std": value_with_std(mape, mape_std),
        "pearson": pearson,
        "pearson_with_std": value_with_std(pearson, pearson_std)
    }

def save_metrics_to_csv(metrics, config, task, result_csv_path=None):
    """
    Save evaluation metrics to a CSV file
    
    Args:
        metrics (dict): Dictionary containing model evaluation metrics
        config (dict): Configuration dictionary with experiment settings
        task (str): Task name 
        result_csv_path (str, optional): Path to save the CSV file. If None, a default path will be created.
    
    Returns:
        str: Path where the CSV was saved
    """    
    exp_name = config.get("exp_name")
    
    if result_csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_dir = os.path.join(base_dir, "csv", exp_name, task)
        os.makedirs(csv_dir, exist_ok=True)
        result_csv_path = os.path.join(csv_dir, f"{exp_name}.csv")
    
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    
    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'exp_name': [config.get("exp_name", "")],
        'mode': [config.get("mode", "")],
        'ring_type': [config.get("dataset", {}).get("ring_type", "")],
        'fold': [config.get("fold", "")],
        'task': [task],
        'input_type': [config.get("dataset", {}).get("input_type", "")],
        'dataset_task': [config.get("dataset", {}).get("task", "")],
        'method_name': [config.get("method", {}).get("name", "")],
        'epochs': [config.get("train", {}).get("epochs", "")],
        'lr': [config.get("train", {}).get("lr", "")],
        'criterion': [config.get("train", {}).get("criterion", "")],
        'batch_size': [config.get("dataset", {}).get("batch_size", "")],
        'sample_len': [metrics['sample_len']],
        'mae_with_std': [metrics['mae_with_std']],
        'rmse_with_std': [metrics['rmse_with_std']],
        'mape_with_std': [metrics['mape_with_std']],
        'pearson_with_std': [metrics['pearson_with_std']]
    })

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(result_csv_path)

    # Write to CSV
    metrics_df.to_csv(result_csv_path, mode='a' if file_exists else 'w', 
              header=not file_exists, index=False)
    
    logging.info(f"Saved results to {result_csv_path}")
    return result_csv_path

def plot_and_save_metrics(predictions, targets, config, task, img_path_folder=None):
    """
    Generate and save visualization plots comparing predictions and targets
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        config (dict): Configuration dictionary with experiment settings
        task (str): Task name to include in the output filenames
        img_path_folder (str, optional): Path to save the generated plots. If None, a default path will be created.
    
    Returns:
        str: Path where the images were saved
    """
    exp_name = config.get("exp_name")
    
    if img_path_folder is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path_folder = os.path.join(base_dir, "img", exp_name)
    
    # Create output directory
    os.makedirs(img_path_folder, exist_ok=True)
    task_img_folder = os.path.join(img_path_folder, task)
    os.makedirs(task_img_folder, exist_ok=True)
    
    # Convert tensors to numpy arrays for plotting
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # 1. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create density-based coloring for points
    xy = np.vstack([target_np, pred_np])
    try:
        z = gaussian_kde(xy)(xy)
        _ = ax.scatter(target_np, pred_np, c=z, s=30, alpha=0.8)
    except np.linalg.LinAlgError:
        # Fallback if KDE fails
        _ = ax.scatter(target_np, pred_np, s=30, alpha=0.5)
    
    # Draw the perfect prediction line
    max_val = max(np.max(pred_np), np.max(target_np))
    min_val = min(np.min(pred_np), np.min(target_np))
    padding = (max_val - min_val) * 0.05
    ax.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 
            '--', color='red', label='Perfect Prediction')
    
    # Add labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predictions vs. Actual Values - {task}')
    
    # Add metrics to plot
    pearson, _ = calculate_pearson(predictions, targets)
    mae, _ = calculate_mae(predictions, targets)
    ax.text(0.05, 0.95, f'Pearson: {pearson:.4f}\nMAE: {mae:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(task_img_folder, f'scatter_plot_{task}.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close(fig)
    
    # 2. Difference plot (Bland-Altman style)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    differences = target_np - pred_np
    averages = (target_np + pred_np) / 2
    
    # Plot with density coloring
    try:
        xy = np.vstack([averages, differences])
        z = gaussian_kde(xy)(xy)
        _ = ax.scatter(averages, differences, c=z, s=30, alpha=0.8)
    except np.linalg.LinAlgError:
        _ = ax.scatter(averages, differences, s=30, alpha=0.5)
    
    # Add mean and confidence intervals
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    ci_upper = mean_diff + 1.96 * std_diff
    ci_lower = mean_diff - 1.96 * std_diff
    
    ax.axhline(mean_diff, color='black', linestyle='-', label=f'Mean: {mean_diff:.4f}')
    ax.axhline(ci_upper, color='red', linestyle='--', label=f'+1.96 SD: {ci_upper:.4f}')
    ax.axhline(ci_lower, color='red', linestyle='--', label=f'-1.96 SD: {ci_lower:.4f}')
    
    ax.set_xlabel('Average of Predicted and Actual')
    ax.set_ylabel('Difference (Actual - Predicted)')
    ax.set_title(f'Bland-Altman Plot - {task}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    diff_path = os.path.join(task_img_folder, f'difference_plot_{task}.png')
    plt.savefig(diff_path, dpi=300)
    plt.close(fig)
    
    # 3. Error distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(differences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(mean_diff, color='red', linestyle='-', label=f'Mean: {mean_diff:.4f}')
    
    ax.set_xlabel('Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution - {task}')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    hist_path = os.path.join(task_img_folder, f'error_histogram_{task}.png')
    plt.savefig(hist_path, dpi=300)
    plt.close(fig)
    
    logging.info(f"Saved visualization plots to {task_img_folder}")
    return task_img_folder
