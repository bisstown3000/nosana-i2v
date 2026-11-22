#!/usr/bin/env bash
# Submit N independent i2v jobs to a Nosana GPU market and collect the job ids.
#
#   scripts/submit_nosana.sh <market> [count]
#
# <market> is a market slug or address, e.g. nvidia-4090 - the market is what
# selects the GPU class, not a field in the job definition.
#
# Requires the Nosana CLI (`npm i -g @nosana/cli`) and a funded wallet.
# Each clip is a separate job on purpose: N jobs on N nodes is the whole point,
# and a failed node costs one clip rather than the batch.
#
# NOTE: re-check the job JSON schema and the flags against the current Nosana
# docs before a paid run (`nosana job post --help`) - an invalid spec still
# consumes the posting fee.
set -euo pipefail

MARKET="${1:?usage: submit_nosana.sh <market> [count]}"
COUNT="${2:-1}"
SPEC="${I2V_SPEC:-nosana/job-i2v.json}"
TIMEOUT_MIN="${I2V_JOB_TIMEOUT_MIN:-60}"
LOG="${I2V_JOB_LOG:-out/jobs.txt}"

if ! command -v nosana >/dev/null 2>&1; then
  echo "nosana CLI not found: npm i -g @nosana/cli" >&2
  exit 1
fi
if grep -q REPLACE_ME "$SPEC"; then
  echo "$SPEC still contains REPLACE_ME - set your pushed image reference first" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
for i in $(seq 1 "$COUNT"); do
  echo "posting job $i/$COUNT to market $MARKET"
  nosana job post \
    --file "$SPEC" \
    --market "$MARKET" \
    --timeout "$TIMEOUT_MIN" \
    | tee -a "$LOG"
done

echo
echo "job ids appended to $LOG"
echo "follow one with:  nosana job get <job-id> --wait"
echo "collect results:  nosana job get <job-id> > out/<node>.log"
echo "                  bench/bench.py out/<node>.log --rate <market-rate>"
echo "(a market job returns logs, not files - the worker prints one"
echo " [i2v-result] line per clip and bench.py reads those lines.)"
