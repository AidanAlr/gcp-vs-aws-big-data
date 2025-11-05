import json
import glob
import numpy as np
import sys
import os

def compare_all_experiments(output_file=None):
    """Print summary stats comparing GCP vs AWS across all experiments"""

    # Redirect output to both console and file if specified
    original_stdout = sys.stdout
    f = None
    if output_file:
        f = open(output_file, 'w')

        class TeeOutput:
            def __init__(self, file_obj, stdout_obj):
                self.file = file_obj
                self.stdout = stdout_obj
            def write(self, text):
                self.stdout.write(text)
                self.file.write(text)
            def flush(self):
                self.stdout.flush()
                self.file.flush()

        sys.stdout = TeeOutput(f, original_stdout)

    experiments = {
        'Storage': 'storage/storage_*.json',
        'PySpark Sample': 'pyspark/pyspark_*_sample_*.json',
        'PySpark Large': 'pyspark/pyspark_*_large_*.json',
        'ML': 'ml/ml_*.json'
    }

    for exp_name, pattern in experiments.items():
        print(f"\n{'='*80}\n{exp_name.upper()}\n{'='*80}")

        if exp_name == 'Storage':
            # Break down storage by operation type
            storage_data = {'aws': {}, 'gcp': {}}
            for file in glob.glob(pattern):
                with open(file) as f:
                    d = json.load(f)
                    provider = d['provider']
                    for op_type, op_data in d['summary'].items():
                        if op_type not in storage_data[provider]:
                            storage_data[provider][op_type] = []
                        storage_data[provider][op_type].append(op_data['avg_duration_seconds'])

            for op_type in ['upload', 'download', 'sequential_read', 'write']:
                print(f"\n{op_type.upper()}:")
                for provider in ['aws', 'gcp']:
                    if op_type in storage_data[provider]:
                        times = np.array(storage_data[provider][op_type])
                        print(f"  {provider.upper()} (n={len(times)}): mean={times.mean():.2f}s  std={times.std():.2f}s  min={times.min():.2f}s  max={times.max():.2f}s")

                if op_type in storage_data['aws'] and op_type in storage_data['gcp']:
                    aws_time = np.mean(storage_data['aws'][op_type])
                    gcp_time = np.mean(storage_data['gcp'][op_type])
                    diff = ((gcp_time - aws_time) / aws_time) * 100
                    print(f"  Comparison: GCP is {diff:+.1f}% vs AWS")
        else:
            data = {'aws': {'times': [], 'costs': []}, 'gcp': {'times': [], 'costs': []}}

            for file in glob.glob(pattern):
                with open(file) as f:
                    d = json.load(f)
                    provider = d['provider']
                    time = d['summary']['total_time_seconds']
                    cost = d['summary']['total_cost_usd']
                    data[provider]['times'].append(time)
                    data[provider]['costs'].append(cost)

            for provider in ['aws', 'gcp']:
                times = np.array(data[provider]['times'])
                costs = np.array(data[provider]['costs'])

                print(f"\n{provider.upper()} (n={len(times)}):")
                print(f"  Time:  mean={times.mean():.2f}s  std={times.std():.2f}s  min={times.min():.2f}s  max={times.max():.2f}s")
                print(f"  Cost:  mean=${costs.mean():.4f}  std=${costs.std():.4f}  min=${costs.min():.4f}  max=${costs.max():.4f}")

            # Comparison
            aws_time, gcp_time = np.mean(data['aws']['times']), np.mean(data['gcp']['times'])
            aws_cost, gcp_cost = np.mean(data['aws']['costs']), np.mean(data['gcp']['costs'])
            time_diff = ((gcp_time - aws_time) / aws_time) * 100
            cost_diff = ((gcp_cost - aws_cost) / aws_cost) * 100
            print(f"\nComparison:")
            print(f"  Time: GCP is {time_diff:+.1f}% vs AWS")
            print(f"  Cost: GCP is {cost_diff:+.1f}% vs AWS")

            # For ML experiments, also load and compare training metrics from cloud_results
            if exp_name == 'ML':
                ml_metrics = {'aws': {'accuracy': [], 'f1_score': []}, 'gcp': {'accuracy': [], 'f1_score': []}}

                # Load metrics from cloud_results organized by provider
                provider_paths = {
                    'aws': 'cloud_results/aws/ml_run*',
                    'gcp': 'cloud_results/gcp_results/ml_run*'
                }

                for provider, path_pattern in provider_paths.items():
                    for ml_dir in glob.glob(path_pattern):
                        metrics_files = glob.glob(os.path.join(ml_dir, 'metrics', 'part-*.txt'))
                        for metrics_file in metrics_files:
                            try:
                                with open(metrics_file) as f:
                                    metrics = json.load(f)
                                    ml_metrics[provider]['accuracy'].append(metrics.get('accuracy', 0))
                                    ml_metrics[provider]['f1_score'].append(metrics.get('f1_score', 0))
                            except Exception as e:
                                pass  # Skip files that can't be parsed

                # Print ML-specific metrics comparison
                if ml_metrics['aws']['accuracy'] and ml_metrics['gcp']['accuracy']:
                    print(f"\nML METRICS:")
                    for provider in ['aws', 'gcp']:
                        accuracy = np.array(ml_metrics[provider]['accuracy'])
                        f1 = np.array(ml_metrics[provider]['f1_score'])
                        print(f"  {provider.upper()} (n={len(accuracy)}):")
                        print(f"    Accuracy: mean={accuracy.mean():.4f}  std={accuracy.std():.4f}  min={accuracy.min():.4f}  max={accuracy.max():.4f}")
                        print(f"    F1 Score: mean={f1.mean():.4f}  std={f1.std():.4f}  min={f1.min():.4f}  max={f1.max():.4f}")

                    # Comparison
                    aws_acc = np.mean(ml_metrics['aws']['accuracy'])
                    gcp_acc = np.mean(ml_metrics['gcp']['accuracy'])
                    aws_f1 = np.mean(ml_metrics['aws']['f1_score'])
                    gcp_f1 = np.mean(ml_metrics['gcp']['f1_score'])
                    acc_diff = ((gcp_acc - aws_acc) / aws_acc) * 100
                    f1_diff = ((gcp_f1 - aws_f1) / aws_f1) * 100
                    print(f"  ML Metrics Comparison:")
                    print(f"    Accuracy: GCP is {acc_diff:+.2f}% vs AWS")
                    print(f"    F1 Score: GCP is {f1_diff:+.2f}% vs AWS")

    # Restore stdout and close file if redirected
    if f:
        sys.stdout = original_stdout
        f.close()

if __name__ == '__main__':
    compare_all_experiments('comparison_summary.txt')
