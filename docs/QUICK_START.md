# Quick Start: Adding Your Workload

## Summary

The infrastructure uses **auto-discovery** - just place your code in the right location and it will be automatically registered.

## Adding a Model/Workload

1. **Create directory**: `src/models/<your_model_name>/`
2. **Create `__init__.py`** with `init_model()` function
3. **Create model file** (e.g., `model.py`) implementing your model
4. **Use it**: `python3 launch.py --model <your_model_name>`

See `src/models/gpt2/` for a complete example.

## Tracking Resource Utilization

A `ResourceUtilStats` class is already implemented that tracks:
- ✅ GPU utilization (%)
- ✅ GPU memory (MB)
- ✅ CPU memory (MB)
- ✅ Disk I/O (read/write MB)

### Usage

```bash
python3 launch.py \
    --model gpt2 \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir ./stats
```

### Output

- **During training**: Statistics printed each step
- **After training**: Summary statistics printed
- **CSV file**: `resource_utilization.csv` saved to output directory

### Example Output

```
GPU Util: 85.3% | GPU Mem: 2048.5 MB | CPU Mem: 512.3 MB | Disk R: 1024.0 MB | Disk W: 256.0 MB
```

## Files Created

- `src/trainer/stats/resource_util.py` - Resource tracking implementation
- `src/config/trainer_stats/resource_util/` - Configuration for resource tracking
- `docs/adding_workloads.md` - Detailed guide

## Next Steps

1. Read `docs/adding_workloads.md` for detailed instructions
2. Check `docs/programming_guide.md` for more advanced features
3. Look at `src/models/gpt2/` as a reference implementation
