import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set modern clean style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Styling definitions
box_style = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2B2D42", linewidth=1.8)
arrow_style = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.0", color="#2B2D42", linewidth=1.8, mutation_scale=15)
curved_arrow = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.25", color="#2B2D42", linewidth=1.8, mutation_scale=15)
dashed_arrow = dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color="#8D99AE", linewidth=1.8, linestyle="--", mutation_scale=15)

# --- TOP ROW: Query Pipeline ---
ax.add_patch(FancyBboxPatch((0.8, 7.8), 3.2, 1.0, **{**box_style, "facecolor": "#EDF2F4"}))
ax.text(2.4, 8.3, "User Input Prompt", ha='center', va='center', fontweight='bold', fontsize=10, color="#2B2D42")

ax.add_patch(FancyBboxPatch((5.0, 7.8), 3.8, 1.0, **{**box_style, "facecolor": "#E8F4F8"}))
ax.text(6.9, 8.3, "Query Understanding Engine\n(Intent, Context & Entity Extractor)", ha='center', va='center', fontweight='bold', fontsize=9, color="#2B2D42")

ax.add_patch(FancyBboxPatch((9.8, 7.8), 3.4, 1.0, **{**box_style, "facecolor": "#D8F3DC"}))
ax.text(11.5, 8.3, "Response Generation\n& Output Calibration", ha='center', va='center', fontweight='bold', fontsize=10, color="#1B4332")

# --- MIDDLE ROW: Memory & Model Core ---
# Memory Layer Box (Left)
ax.add_patch(FancyBboxPatch((0.8, 4.0), 3.8, 3.0, **{**box_style, "facecolor": "#F0F3F8"}))
ax.text(2.7, 6.6, "Tiered Memory Core", ha='center', va='center', fontweight='bold', fontsize=11, color="#1D3557")
ax.text(2.7, 6.0, "• Ephemeral, Working & Long-Term", ha='center', va='center', fontsize=8.5, color="#2B2D42")
ax.text(2.7, 5.5, "• Vector DB (Qdrant Embeddings)", ha='center', va='center', fontsize=8.5, color="#2B2D42")
ax.text(2.7, 5.0, "• Ebbinghaus Decay & RIF Suppression", ha='center', va='center', fontsize=8.5, color="#2B2D42")
ax.text(2.7, 4.5, "• Dynamic Knowledge Graph", ha='center', va='center', fontsize=8.5, color="#2B2D42")

# Real-Time API Box (Center)
ax.add_patch(FancyBboxPatch((5.4, 5.7), 3.6, 1.2, **{**box_style, "facecolor": "#FFF3BF"}))
ax.text(7.2, 6.3, "Real-Time Web Fetcher", ha='center', va='center', fontweight='bold', fontsize=10, color="#744210")
ax.text(7.2, 5.9, "Google Custom Search & Live API", ha='center', va='center', fontsize=8.5, color="#744210")

# LLM Core Box (Center-Right)
ax.add_patch(FancyBboxPatch((5.4, 4.0), 3.6, 1.3, **{**box_style, "facecolor": "#EBE4F9"}))
ax.text(7.2, 4.8, "Base LLM Core Engine", ha='center', va='center', fontweight='bold', fontsize=10, color="#4A154B")
ax.text(7.2, 4.3, "Mistral-7B / LLaMA with LoRA Adapters", ha='center', va='center', fontsize=8.5, color="#4A154B")

# --- BOTTOM ROW: Continual Learning & Fact Checking ---
# Async Validation
ax.add_patch(FancyBboxPatch((0.8, 0.8), 3.8, 2.2, **{**box_style, "facecolor": "#FFE8D6"}))
ax.text(2.7, 2.5, "Async Post-Validation Pipeline", ha='center', va='center', fontweight='bold', fontsize=10.5, color="#6B2D5C")
ax.text(2.7, 1.9, "• Fact Verification Guardrails", ha='center', va='center', fontsize=8.5, color="#2B2D42")
ax.text(2.7, 1.4, "• Hallucination Mitigation Check", ha='center', va='center', fontsize=8.5, color="#2B2D42")

# Experience Buffer
ax.add_patch(FancyBboxPatch((5.2, 0.8), 3.8, 2.2, **{**box_style, "facecolor": "#E2F0D9"}))
ax.text(7.1, 2.5, "Consolidation Experience Buffer", ha='center', va='center', fontweight='bold', fontsize=10.5, color="#1E4620")
ax.text(7.1, 1.9, "• Verified Claims & Flagged Pairs", ha='center', va='center', fontsize=8.5, color="#1E4620")
ax.text(7.1, 1.4, "• High-Confidence Feedback Storage", ha='center', va='center', fontsize=8.5, color="#1E4620")

# LoRA Trainer
ax.add_patch(FancyBboxPatch((9.6, 0.8), 3.6, 2.2, **{**box_style, "facecolor": "#F3E5F5"}))
ax.text(11.4, 2.5, "Continual LoRA Trainer", ha='center', va='center', fontweight='bold', fontsize=10.5, color="#4A148C")
ax.text(11.4, 1.9, "• Sleep Learning & Meta-Learning", ha='center', va='center', fontsize=8.5, color="#4A148C")
ax.text(11.4, 1.4, "• Catastrophic Forgetting Prevention", ha='center', va='center', fontsize=8.5, color="#4A148C")

# --- CONNECTING ARROWS ---
# Top Pipeline Flow
ax.annotate("", xy=(5.0, 8.3), xytext=(4.0, 8.3), arrowprops=arrow_style)
ax.annotate("", xy=(9.8, 8.3), xytext=(8.8, 8.3), arrowprops=arrow_style)

# Query to Memory & Fetcher
ax.annotate("", xy=(2.7, 7.0), xytext=(2.4, 7.8), arrowprops=curved_arrow)
ax.annotate("", xy=(7.2, 6.9), xytext=(6.9, 7.8), arrowprops=arrow_style)

# Memory & Fetcher to LLM Core
ax.annotate("", xy=(5.4, 4.65), xytext=(4.6, 4.65), arrowprops=arrow_style)
ax.annotate("", xy=(7.2, 5.3), xytext=(7.2, 5.7), arrowprops=arrow_style)

# LLM Core to Response Generation
ax.annotate("", xy=(11.5, 7.8), xytext=(8.8, 5.0), arrowprops=curved_arrow)

# Response to Async Fact Check (Dashed)
ax.annotate("", xy=(2.7, 3.0), xytext=(11.5, 7.8), arrowprops=dashed_arrow)
ax.text(8.0, 3.6, "Async Validation Stream", ha='center', va='center', fontsize=8.5, color="#6C757D", style="italic")

# Bottom Pipeline Flow
ax.annotate("", xy=(5.2, 1.9), xytext=(4.6, 1.9), arrowprops=arrow_style)
ax.annotate("", xy=(9.6, 1.9), xytext=(9.0, 1.9), arrowprops=arrow_style)
ax.annotate("", xy=(7.2, 4.0), xytext=(11.4, 3.0), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color="#4A148C", linewidth=1.5, linestyle="--", mutation_scale=15))

# Header & Branding Title
ax.text(7.0, 9.6, "LP-LLM Architecture: Self-Evolving Cognitive System", ha='center', va='center', fontweight='bold', fontsize=14, color="#1D3557")
ax.text(7.0, 9.25, "Authored by Shuvam (https://github.com/samshuvam)", ha='center', va='center', fontsize=9.5, color="#457B9D", style="italic")

plt.tight_layout()

# Save images both in project root and images/ directory
output_dirs = [".", "images"]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)
    plt.savefig(os.path.join(d, 'architecture_flow.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(d, 'architecture_flow.pdf'), bbox_inches='tight')

print("✓ Successfully regenerated sleek LP-LLM Architecture Flow Diagram by Shuvam!")