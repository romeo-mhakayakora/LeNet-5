# Running without install

If you do not install the package, set `PYTHONPATH` so `src/` is importable.

PowerShell:
```powershell
$env:PYTHONPATH = (Resolve-Path .\\src)
python .\\scripts\\train.py
```

