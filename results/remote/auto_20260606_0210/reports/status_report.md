# TGPE v4 实验状态报告

> 自动收集时间: 2026-06-06 02:19:45 CST
> 本地路径: /home/xieguiawu/Desktop/ML/Glaubenskrieg/results/remote/auto_20260606_0210

## 远程进程状态

### Server 2 (223.109.239.32:20248)
S2_GPU_PLACEHOLDER

```
none
```

### Server 1 (223.109.239.36:24520)
S1_GPU_PLACEHOLDER

S1_PROCS_PLACEHOLDER

## 下载文件清单

```
FILELIST_PLACEHOLDER
```

## 快速分析命令

```bash
cd /home/xieguiawu/Desktop/ML/Glaubenskrieg/results/remote/auto_20260606_0210
python3 -c "
import json, glob
for f in sorted(glob.glob('s*_results/*.json')):
    d = json.load(open(f))
    nm = f.split('/')[-1]
    ts = d.get('test_metrics',{}).get('test_sharpe','N/A')
    ic = d.get('mean_ctm_ic', d.get('mean_ensemble_ic', 'N/A'))
    print(f'{nm}: test_sharpe={ts}, IC={ic}')
"
```
