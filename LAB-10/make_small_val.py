from pathlib import Path
import random
import yaml
import ultralytics
from ultralytics.utils import SETTINGS

here = Path(__file__).resolve().parent
src = Path(ultralytics.__file__).parent / 'cfg' / 'datasets' / 'VOC.yaml'
cfg = yaml.safe_load(src.read_text(encoding='utf-8'))

# หาโฟลเดอร์ VOC ที่โหลดไว้ (แบบเดียวกับที่ Ultralytics หา)
voc_root = (Path(SETTINGS['datasets_dir']) / cfg['path']).resolve()
img_dir = voc_root / 'images' / 'test2007'
if not img_dir.exists():
    raise FileNotFoundError(f'ไม่พบ {img_dir}  (ตรวจว่า VOC โหลดเสร็จแล้ว)')

# สุ่มภาพ val/test 500 ภาพ
imgs = sorted(img_dir.glob('*.jpg'))
random.seed(0)
subset = random.sample(imgs, min(500, len(imgs)))
val_txt = here / 'val_small.txt'
val_txt.write_text('\n'.join(p.as_posix() for p in subset), encoding='utf-8')

# สร้าง VOC_small.yaml
cfg['path'] = voc_root.as_posix()
cfg['val'] = val_txt.as_posix()
cfg['test'] = val_txt.as_posix()
cfg.pop('download', None)
out = here / 'VOC_small.yaml'
out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

print(f'VOC: {voc_root}')
print(f'ภาพทั้งหมด {len(imgs)} -> ใช้ {len(subset)} ภาพ')
print(f'สร้างแล้ว: {val_txt.name}, {out.name}')