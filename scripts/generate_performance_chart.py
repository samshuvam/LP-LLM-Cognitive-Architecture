import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set modern clean style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9.5

# Data from experimental results
metrics = ['Latency (s)', 'Hallucination (%)', 'Retention @ 7d (%)', 'Query Acc. (%)']
LP_LLM = [3.21, 8.7, 82.4, 86.9]
baseline_rag = [5.87, 24.3, 45.2, 72.1]
static_llm = [2.14, 31.5, 12.1, 68.3]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars1 = ax.bar(x - width, LP_LLM, width, label='LP-LLM (Ours)', color='#2E86AB', edgecolor='#1D3557', linewidth=0.8)
bars2 = ax.bar(x, baseline_rag, width, label='Vanilla RAG', color='#A23B72', edgecolor='#4A154B', linewidth=0.8)
bars3 = ax.bar(x + width, static_llm, width, label='Static LLM', color='#F18F01', edgecolor='#B05D00', linewidth=0.8)

# Add error bars
ax.errorbar(x - width, LP_LLM, yerr=[0.45, 1.2, 3.1, 2.1], fmt='none', color='black', capsize=3, linewidth=1)
ax.errorbar(x, baseline_rag, yerr=[1.23, 3.1, 5.6, 3.4], fmt='none', color='black', capsize=3, linewidth=1)
ax.errorbar(x + width, static_llm, yerr=[0.31, 4.2, 2.3, 4.1], fmt='none', color='black', capsize=3, linewidth=1)

ax.set_ylabel('Performance Metric Value', fontweight='bold', color='#1D3557')
ax.set_title('LP-LLM Performance Comparison with Baselines (Shuvam)', fontweight='bold', pad=15, fontsize=12, color='#1D3557')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
ax.grid(True, linestyle=':', alpha=0.6)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()

output_dirs = [".", "images"]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)
    plt.savefig(os.path.join(d, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(d, 'performance_comparison.pdf'), bbox_inches='tight')

print("✓ Successfully regenerated LP-LLM Performance Comparison Chart!")