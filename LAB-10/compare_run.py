from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'runs' / 'detect' / 'runs'

# ชื่อโฟลเดอร์ของแต่ละรอบ (แก้ได้ เรียงตามจำนวน epoch)
RUNS = ['voc_e5', 'voc_e10', 'voc_e15']

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
rows = []

for i, name in enumerate(RUNS):
    csv = BASE / name / 'results.csv'
    if not csv.exists():
        print(f'ข้าม {name}: ไม่พบ {csv}')
        continue
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()   # ชื่อคอลัมน์มีช่องว่างนำหน้า
    c = colors[i % len(colors)]
    ep = df['epoch']
    n = len(df)

    # กราฟซ้าย: mAP (ทึบ = mAP50-95, ประ = mAP50)
    ax1.plot(ep, df['metrics/mAP50-95(B)'], color=c, label=f'mAP50-95 ({n} ep)')
    ax1.plot(ep, df['metrics/mAP50(B)'], color=c, linestyle='--', label=f'mAP50 ({n} ep)')

    # กราฟขวา: loss รวม (ทึบ = train, ประ = val)
    train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    ax2.plot(ep, train_loss, color=c, label=f'Train ({n} ep)')
    ax2.plot(ep, val_loss, color=c, linestyle='--', label=f'Val ({n} ep)')

    last = df.iloc[-1]
    rows.append({
        'run': name,
        'epochs': n,
        'precision': round(last['metrics/precision(B)'], 4),
        'recall': round(last['metrics/recall(B)'], 4),
        'mAP50': round(last['metrics/mAP50(B)'], 4),
        'mAP50-95': round(last['metrics/mAP50-95(B)'], 4),
        'best_mAP50-95': round(df['metrics/mAP50-95(B)'].max(), 4),
        'final_train_loss': round(train_loss.iloc[-1], 4),
        'final_val_loss': round(val_loss.iloc[-1], 4),
    })

if not rows:
    raise SystemExit('ไม่พบไฟล์ results.csv เลย ตรวจชื่อรอบและ path')

ax1.set_title('Model mAP Comparison (Validation)')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('mAP')
ax2.set_title('Model Loss Comparison')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig(ROOT / 'compare_runs.png', dpi=150)

summary = pd.DataFrame(rows)
summary.to_csv(ROOT / 'compare_summary.csv', index=False)
print(summary.to_string(index=False))
print('บันทึกแล้ว: compare_runs.png, compare_summary.csv')