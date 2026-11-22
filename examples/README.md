# examples

`job.json` is a ready-to-run two-clip batch. Drop the two source stills next to
it as `hook_01.png` and `hook_02.png` (any resolution — the worker resizes to
`I2V_WIDTH`×`I2V_HEIGHT`), then:

    scripts/run_local.sh examples/ out/

Stills are not committed: image-to-video output inherits whatever is in the
source frame, so each user should supply their own licence-clean input.
