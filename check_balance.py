import torch, glob

step = 212
layer = 3

files = glob.glob(f"/local/home/bingli/moe/traces/mirror-moe-38m/actives/{step}/{layer}/*.pt")
counts = torch.zeros(64, dtype=torch.long)

for f in files:
    values, indices = torch.load(f, map_location="cpu")  # [tokens, topk=6]
    counts += torch.bincount(indices.view(-1), minlength=64)

counts_f = counts.float() / counts.sum()
print(f"ideal={1/64:.4f}  max={counts_f.max():.4f}  min={counts_f.min():.4f}  std={counts_f.std():.4f}")
