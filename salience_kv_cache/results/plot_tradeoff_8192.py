import matplotlib.pyplot as plt
import json

# Load data
with open('tradeoff_data_8192.json', 'r') as f:
    data = json.load(f)

tau = data['tau_values']
compression = data['compression_ratios']
quality = data['quality_degradation']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Compression vs Quality
ax1.plot(compression, quality, 'o-', linewidth=2, markersize=6, label='Tiered Compression')
ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Sweet Spot Threshold (0.5%)')
ax1.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Cliff Threshold (5%)')
ax1.axvline(x=7.360287511230908, 
            color='g', linestyle=':', alpha=0.5)
ax1.fill_between(compression, 0, 0.5, alpha=0.2, color='green', label='Sweet Spot')
ax1.set_xlabel('Compression Ratio (x)', fontsize=12)
ax1.set_ylabel('Quality Degradation (%)', fontsize=12)
ax1.set_title('Compression vs Quality Tradeoff', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 10)

# Plot 2: Tau sweep
ax2.plot(tau, compression, 'o-', linewidth=2, markersize=6, label='Compression Ratio', color='blue')
ax2_twin = ax2.twinx()
ax2_twin.plot(tau, quality, 's-', linewidth=2, markersize=6, label='Quality Loss', color='red')
ax2.set_xlabel('τ (Retention Threshold)', fontsize=12)
ax2.set_ylabel('Compression Ratio (x)', fontsize=12, color='blue')
ax2_twin.set_ylabel('Quality Degradation (%)', fontsize=12, color='red')
ax2.set_title('Effect of Threshold on Compression & Quality', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tradeoff_curves_8192.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to tradeoff_curves_8192.png")
