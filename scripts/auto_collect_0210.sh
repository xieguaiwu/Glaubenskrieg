#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# auto_collect_0210.sh — 定时实验数据收集脚本
#
# 运行窗口: 2026-06-06 02:10 ~ 02:20 CST
# 每 60s 轮询一次远程服务器状态，下载结果和日志。
# ═══════════════════════════════════════════════════════════════════════════
# 确保 at/cron 环境下 PATH 正确
export PATH="/usr/local/bin:/usr/bin:/bin:/home/xieguiawu/.local/bin:$PATH"
set -o pipefail

# ── 配置 ────────────────────────────────────────────────────────────────
SSHPASS_S2="iewoh9vu"
SSHPASS_S1="The9phae"
HOST_S2="223.109.239.32"
PORT_S2="20248"
HOST_S1="223.109.239.36"
PORT_S1="24520"
USER="root"

LOCAL_BASE="/home/xieguiawu/Desktop/ML/Glaubenskrieg/results/remote/auto_$(date +%Y%m%d_%H%M)"
REMOTE_RESULTS_DIR="/root/results/tgpe_v4"
REMOTE_LOGS_DIR="/root/logs"

SSH_S2=(sshpass -p "$SSHPASS_S2" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p "$PORT_S2" "$USER@$HOST_S2")
SSH_S1=(sshpass -p "$SSHPASS_S1" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p "$PORT_S1" "$USER@$HOST_S1")
SCP_S2=(sshpass -p "$SSHPASS_S2" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 -P "$PORT_S2")
SCP_S1=(sshpass -p "$SSHPASS_S1" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 -P "$PORT_S1")

DEADLINE_EPOCH=$(TZ=Asia/Shanghai date -d "2026-06-06 02:20:00" +%s 2>/dev/null || echo 0)

mkdir -p "$LOCAL_BASE"/{s2_results,s1_results,s2_logs,s1_logs,reports} || { echo "FATAL: cannot create output dirs"; exit 1; }
LOGFILE="$LOCAL_BASE/collect.log"
exec > >(tee -a "$LOGFILE") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 工具函数 ────────────────────────────────────────────────────────────

# Escape sed special characters (&, \, and the delimiter in use)
escape_sed() { printf '%s' "$1" | sed 's/[&/\\]/\\&/g'; }

remote_cmd() {
    local server="$1"; shift
    if [ "$server" = "s2" ]; then
        "${SSH_S2[@]}" "$@" 2>/dev/null
    else
        "${SSH_S1[@]}" "$@" 2>/dev/null
    fi
}

download_results() {
    local server="$1" dest="$2"
    local scp_cmd host port pw
    if [ "$server" = "s2" ]; then
        pw="$SSHPASS_S2"; host="$HOST_S2"; port="$PORT_S2"
        sshpass -p "$pw" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
            -P "$port" "${USER}@${host}:${REMOTE_RESULTS_DIR}/*.json" "$dest/" 2>/dev/null
    else
        pw="$SSHPASS_S1"; host="$HOST_S1"; port="$PORT_S1"
        sshpass -p "$pw" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
            -P "$port" "${USER}@${host}:${REMOTE_RESULTS_DIR}/*.json" "$dest/" 2>/dev/null
    fi
}

download_logs() {
    local server="$1" dest="$2"
    local pw host port
    if [ "$server" = "s2" ]; then
        pw="$SSHPASS_S2"; host="$HOST_S2"; port="$PORT_S2"
    else
        pw="$SSHPASS_S1"; host="$HOST_S1"; port="$PORT_S1"
    fi
    sshpass -p "$pw" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
        -P "$port" "${USER}@${host}:${REMOTE_LOGS_DIR}/tgpe_v4_*.log" "$dest/" 2>/dev/null
}

is_deadline_passed() { [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; }

# ── 状态检查 ────────────────────────────────────────────────────────────

check_server() {
    local label="$1" server="$2"
    log "── $label ──"

    # GPU
    local gpu
    gpu=$(remote_cmd "$server" 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader' 2>/dev/null)
    if [ -n "$gpu" ]; then
        log "  GPU: $(echo "$gpu" | tr '\n' ' | ')"
    else
        log "  GPU: unreachable"
    fi

    # Train processes
    local procs
    procs=$(remote_cmd "$server" 'ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E "train\.py" | grep -v grep' 2>/dev/null)
    if [ -n "$procs" ]; then
        count=$(printf '%s' "$procs" | wc -l)
        log "  train.py processes: $count"
        printf '%s\n' "$procs" | while read -r line; do
            log "    $line"
        done
    else
        log "  train.py: none running"
    fi

    # SMT processes
    local smt
    smt=$(remote_cmd "$server" 'ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E "train_200k|smt" | grep -v grep' 2>/dev/null)
    if [ -n "$smt" ]; then
        log "  SMT processes:"
        printf '%s\n' "$smt" | while read -r line; do
            log "    $line"
        done
    fi

    # Result files
    local results
    results=$(remote_cmd "$server" "ls -la ${REMOTE_RESULTS_DIR}/*.json 2>/dev/null" 2>/dev/null)
    if [ -n "$results" ]; then
        log "  Results:"
        printf '%s\n' "$results" | while read -r line; do
            log "    $line"
        done
    else
        log "  Results: none yet"
    fi

    # Latest log tail
    local latest_log
    latest_log=$(remote_cmd "$server" "ls -t ${REMOTE_LOGS_DIR}/tgpe_v4_*.log 2>/dev/null | head -1" 2>/dev/null)
    if [ -n "$latest_log" ]; then
        log "  Latest log tail ($latest_log):"
        remote_cmd "$server" "tail -3 \"${latest_log}\" 2>/dev/null" | while read -r line; do
            log "    $line"
        done
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# 主循环: 02:10 ~ 02:20
# ═══════════════════════════════════════════════════════════════════════════

log "══════════ AUTO COLLECT STARTED ══════════"
log "Local dir: $LOCAL_BASE"
log "Window: 02:10 ~ 02:20 CST"
log ""

CYCLE=0

while ! is_deadline_passed; do
    CYCLE=$((CYCLE + 1))
    log ""
    log "╔══ CYCLE $CYCLE ═══════════════════════════════════════"
    log "║ Time: $(date '+%H:%M:%S'), ${DEADLINE_EPOCH} deadline, now $(date +%s)"

    # ── Status checks ──
    check_server "Server 2 (32:20248)" "s2"
    echo ""
    check_server "Server 1 (36:24520)" "s1"

    # ── Download results ──
    log ""
    log "── Downloading results ──"

    log "  S2 → $LOCAL_BASE/s2_results/"
    download_results "s2" "$LOCAL_BASE/s2_results"
    log "  S1 → $LOCAL_BASE/s1_results/"
    download_results "s1" "$LOCAL_BASE/s1_results"

    # ── Download logs ──
    log "  S2 logs → $LOCAL_BASE/s2_logs/"
    download_logs "s2" "$LOCAL_BASE/s2_logs"
    log "  S1 logs → $LOCAL_BASE/s1_logs/"
    download_logs "s1" "$LOCAL_BASE/s1_logs"

    # ── Local summary of what we have ──
    local_count=$(find "$LOCAL_BASE" -name "*.json" -type f 2>/dev/null | wc -l)
    log "  Local JSON files collected so far: $local_count"

    # Dynamic sleep: min(60, remain-55) to avoid overshooting deadline
    if ! is_deadline_passed; then
        remain=$((DEADLINE_EPOCH - $(date +%s)))
        if [ "$remain" -le 55 ]; then
            log "  < 55s to deadline, doing final collection..."
            break
        fi
        sleep_time=$(( remain > 115 ? 60 : remain - 55 ))
        [ "$sleep_time" -gt 0 ] && { log "  Sleeping ${sleep_time}s... (${remain}s remaining)"; sleep "$sleep_time"; }
    fi
done

# ═══════════════════════════════════════════════════════════════════════════
# 最终收集 (02:20 之后)
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "══════════ FINAL COLLECTION ══════════"
log "Time: $(date)"

# Final status
check_server "Server 2 (32:20248) FINAL" "s2"
echo ""
check_server "Server 1 (36:24520) FINAL" "s1"

# Final download (force)
log ""
log "── Final download ──"
download_results "s2" "$LOCAL_BASE/s2_results"
download_results "s1" "$LOCAL_BASE/s1_results"
download_logs "s2" "$LOCAL_BASE/s2_logs"
download_logs "s1" "$LOCAL_BASE/s1_logs"

# Also download SMT results if any
log "── SMT results ──"
sshpass -p "$SSHPASS_S2" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    -P "$PORT_S2" "${USER}@${HOST_S2}:/root/smt_model/model/*.json" "$LOCAL_BASE/s2_results/" 2>/dev/null && log "  S2 SMT: downloaded" || log "  S2 SMT: none"
sshpass -p "$SSHPASS_S1" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    -P "$PORT_S1" "${USER}@${HOST_S1}:/root/smt_model/model/*.json" "$LOCAL_BASE/s1_results/" 2>/dev/null && log "  S1 SMT: downloaded" || log "  S1 SMT: none"

# ── 生成报告 ──

REPORT="$LOCAL_BASE/reports/status_report.md"
cat > "$REPORT" << 'REPORTEOF'
# TGPE v4 实验状态报告

> 自动收集时间: COLLECT_TIME
> 本地路径: LOCAL_BASE

## 远程进程状态

### Server 2 (223.109.239.32:20248)
S2_GPU_PLACEHOLDER

S2_PROCS_PLACEHOLDER

### Server 1 (223.109.239.36:24520)
S1_GPU_PLACEHOLDER

S1_PROCS_PLACEHOLDER

## 下载文件清单

```
FILELIST_PLACEHOLDER
```

## 快速分析命令

```bash
cd LOCAL_BASE
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
REPORTEOF

# Fill placeholders (with sed special-char escaping)
COLLECT_TIME="$(date '+%Y-%m-%d %H:%M:%S CST')"
sed -i "s|COLLECT_TIME|$(escape_sed "$COLLECT_TIME")|g" "$REPORT"
sed -i "s|LOCAL_BASE|$(escape_sed "$LOCAL_BASE")|g" "$REPORT"

# GPU placeholders
S2_GPU=$(remote_cmd "s2" 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader' 2>/dev/null || echo "unreachable")
S1_GPU=$(remote_cmd "s1" 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader' 2>/dev/null || echo "unreachable")
sed -i "s|S2_GPU_PLACEHOLDER|\`\`\`\n$(escape_sed "$S2_GPU")\n\`\`\`|g" "$REPORT"
sed -i "s|S1_GPU_PLACEHOLDER|\`\`\`\n$(escape_sed "$S1_GPU")\n\`\`\`|g" "$REPORT"

# Process placeholders
S2_PROCS=$(remote_cmd "s2" 'ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E "train\.py|train_200k" | grep -v grep' 2>/dev/null || echo "none")
S1_PROCS=$(remote_cmd "s1" 'ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E "train\.py|train_200k" | grep -v grep' 2>/dev/null || echo "none")
sed -i "s|S2_PROCS_PLACEHOLDER|\`\`\`\n$(escape_sed "$S2_PROCS")\n\`\`\`|g" "$REPORT"
sed -i "s|S1_PROCS_PLACEHOLDER|\`\`\`\n$(escape_sed "$S1_PROCS")\n\`\`\`|g" "$REPORT"

# File list
FILELIST=$(find "$LOCAL_BASE" -type f \( -name "*.json" -o -name "*.log" \) -exec ls -lh {} \; 2>/dev/null | sort)
sed -i "s|FILELIST_PLACEHOLDER|\`\`\`\n$(escape_sed "$FILELIST")\n\`\`\`|g" "$REPORT"

log ""
log "══════════ DONE ══════════"
log "Report: $REPORT"
log "Data:   $LOCAL_BASE"
log ""

# Print report to stdout for at/cron mail
cat "$REPORT"
