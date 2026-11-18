#!/usr/bin/env bash
# Dead-man watch for the login1 elastic controller (2026-11-13). Runs on a cpu node, so it survives
# login1 losing /proj or rebooting (06:48 that day: the controller died silently and every run,
# SA3 and LiveBand, sat dead ~8 h). If the controller heartbeat is >10 min old it alerts (ntfy,
# hourly) and ticks every run itself until the controller is back. It keeps one singleton
# successor queued; the controller resubmits it if none is queued. Stop: touch $ROOT/.watchdog_off.
#SBATCH -A slurm_grp_preemptable
#SBATCH -p cpu
#SBATCH --qos=cpu
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 1-00:00:00
#SBATCH -J el:watchdog
#SBATCH -o /proj/spans/brade/lmdm/runs/_watchdog/%j.out
#SBATCH --open-mode=append
# CSAIL (2026-11-15): the controller submits this with site flags (-A/-p/--qos/-o) and exports
# ELASTIC_SITE / ELASTIC_HOME / ELASTIC_PY; unset, the Blue Vela values below apply unchanged.
exec ${ELASTIC_PY:-/usr/bin/python3} -I ${ELASTIC_HOME:-/u/brade/live-music-diffusion-models-sa3/elastic}/elastic.py watchdog \
    --interval 120 --stale 600 --hours 23.5
