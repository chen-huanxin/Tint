# Deep Context Fusion Network

## create environment
```bash
conda env create -f typhoon-intensity.yaml
conda activate typhoon-intensity
```

## train
```bash
python train.py --gpu=X --model=XXX
```

## test

```bash
python test.py --gpu=X [--smooth] --weights="模型路径"
```

