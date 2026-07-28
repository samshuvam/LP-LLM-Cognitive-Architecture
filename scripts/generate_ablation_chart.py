import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set modern clean style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9.5

# Data from ablation study
configurations = ['Full LP-LLM', 'Without RIF', 'Without Validation', 'Without Meta-Learning', 'Without Decay', 'Without LoRA']
retention = [82.4, 74.2, 81.9, 79.1, 68.4, 45.2]
hallucination = [8.7, 9.1, 18.3, 10.2, 11.5, 24.3]

x = np.arange(len(configurations))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, dpi=300)

# Top subplot: Memory Retention
bars1 = ax1.bar(x, retention, width, color='#2E86AB', edgecolor='#1D3557', linewidth=0.8)
ax1.set_ylabel('Memory Retention (%)', fontweight='bold', color='#1D3557')
ax1.set_title('LP-LLM Ablation Study: Impact of Cognitive Components (Shuvam)', fontweight='bold', pad=15, fontsize=12, color='#1D3557')
ax1.axhline(y=82.4, color='#E63946', linestyle='--', alpha=0.8, linewidth=1.2, label='Full LP-LLM (82.4%)')
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
ax1.grid(True, linestyle=':', alpha=0.6)

# Bottom subplot: Hallucination Rate
bars2 = ax2.bar(x, hallucination, width, color='#A23B72', edgecolor='#4A154B', linewidth=0.8)
ax2.set_ylabel('Hallucination Rate (%)', fontweight='bold', color='#4A154B')
ax2.set_xticks(x)
ax2.set_xticklabels(configurations, rotation=25, ha='right', fontweight='medium')
ax2.axhline(y=8.7, color='#E63946', linestyle='--', alpha=0.8, linewidth=1.2, label='Full LP-LLM (8.7%)')
ax2.set_ylim(0, 30)
ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
ax2.grid(True, linestyle=':', alpha=0.6)

# Add value labels
for ax, bars in [(ax1, bars1), (ax2, bars2)]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()

output_dirs = [".", "images"]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)
    plt.savefig(os.path.join(d, 'ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(d, 'ablation_study.pdf'), bbox_inches='tight')

print("✓ Successfully regenerated LP-LLM Ablation Study Chart!")