#!/bin/bash
# Run the DeepECG-Ancestry pipeline from ancestry.config, the same way run_pipeline.bash runs
# the main pipeline from heartwise.config.
#
#   bash run_ancestry.bash                    # the mode in ancestry.config
#   bash run_ancestry.bash --mode survey      # read-only: what is on this machine
#   bash run_ancestry.bash --mode selftest    # a synthetic cohort, end to end
#   bash run_ancestry.bash --mode all         # the whole study
#
# Any further arguments are passed through to the mode, so a one-off stays a one-off:
#   bash run_ancestry.bash --mode loci --threads 32
#
# Run it with `bash`, not `source`. run_pipeline.bash returns at the top level, which works only
# when sourced; this one exits, so `bash run_ancestry.bash` reports a real exit status -- which
# matters here because the whole study is one command and a wrapper that always exits 0 would
# hide a stage that failed.

usage() {
    echo "Usage: bash run_ancestry.bash [--mode <mode>] [extra arguments ...]"
    echo "  --mode   survey | selftest | weights | convert | prepare | score | traits |"
    echo "           geno | loci | pgs | scan |"
    echo "           fairness | ses | figures | package | submit | validate | metrics |"
    echo "           admixed | ensemble | all   (default: the mode in ancestry.config)"
    echo "  everything else is forwarded to that mode; <mode> --help lists its options"
    return 1
}

get_param() {
    grep "^$1:" ancestry.config | cut -d':' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

mode=$(get_param "mode")
device=$(get_param "device")
batch_size=$(get_param "batch_size")
threads=$(get_param "threads")
jobs=$(get_param "jobs")
work_folder=$(get_param "work_folder")
results_folder=$(get_param "results_folder")
run_name=$(get_param "run_name")
rfmix=$(get_param "rfmix")
ecg_meta=$(get_param "ecg_meta")
geno=$(get_param "geno")
build=$(get_param "build")
labels=$(get_param "labels")
statements=$(get_param "statements")
measurements=$(get_param "measurements")
ses=$(get_param "ses")
with_scan=$(get_param "with_scan")
save_embeddings=$(get_param "save_embeddings")
ancestry_model_repo=$(get_param "ancestry_model_repo")
ancestry_model_revision=$(get_param "ancestry_model_revision")
weights_source=$(get_param "weights_source")

# The snapshot's repository and revision reach the package as environment variables, which is
# also how a site that prefers `docker run -e` gives them. Empty means "use the manifest".
[[ -n "$ancestry_model_repo" ]]     && export ANCESTRY_MODEL_REPO="$ancestry_model_repo"
[[ -n "$ancestry_model_revision" ]] && export ANCESTRY_MODEL_REVISION="$ancestry_model_revision"
[[ -n "$weights_source" ]]          && export ANCESTRY_WEIGHTS_SOURCE="$weights_source"

extra=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then mode="$2"; shift 2
            else echo "Error: --mode requires a non-empty argument."; usage; exit 1; fi
            ;;
        --help|-h) usage; exit 0 ;;
        *) extra+=("$1"); shift ;;
    esac
done

# `survey`, `selftest` and `weights` take no study paths and no --work/--results: they are what
# a site runs before it has any.
case $mode in
    survey)
        exec python -m ancestry survey --rfmix $rfmix --ecg-meta $ecg_meta \
             --geno "$geno" "${extra[@]}"
        ;;
    selftest)
        exec python -m ancestry selftest --out "$work_folder/selftest" "${extra[@]}"
        ;;
    weights)
        exec python -m ancestry weights "${extra[@]}"
        ;;
esac

args=(--work "$work_folder" --results "$results_folder")
case $mode in
    all)      args+=(--run-name "$run_name" --device "$device" --batch-size "$batch_size"
                     --threads "$threads" --jobs "$jobs" --build "$build"
                     --rfmix $rfmix --ecg-meta $ecg_meta --geno "$geno")
              [[ -n $labels ]]       && args+=(--labels "$labels")
              [[ -n $statements ]]   && args+=(--statements "$statements")
              [[ -n $measurements ]] && args+=(--measurements "$measurements")
              [[ -n $ses ]]          && args+=(--ses "$ses")
              [[ $with_scan == true ]]       && args+=(--with-scan)
              [[ $save_embeddings == true ]] && args+=(--save-embeddings)
              ;;
    prepare)  args+=(--rfmix $rfmix --ecg-meta $ecg_meta) ;;
    score)    args+=(--device "$device" --batch-size "$batch_size")
              [[ $save_embeddings == true ]] && args+=(--save-embeddings) ;;
    geno)     args+=(--geno "$geno" --build "$build" --threads "$threads") ;;
    scan)     args+=(--geno "$geno" --threads "$threads") ;;
    loci|pgs) args+=(--threads "$threads") ;;
    traits)   args+=(--jobs "$jobs") ;;
    fairness) [[ -n $labels ]] && args+=(--labels "$labels")
              [[ -n $statements ]] && args+=(--statements "$statements") ;;
    ses)      args+=(--ses "$ses") ;;
    package)  args+=(--run-name "$run_name") ;;
    figures)  args=(--results "$results_folder") ;;
    convert)  args+=(--vcf "$geno") ;;
    submit|validate|metrics|admixed|ensemble) : ;;
    *) echo "Invalid mode: $mode"; usage; exit 1 ;;
esac

echo "Running the ancestry pipeline in $mode mode..."
exec python -m ancestry "$mode" "${args[@]}" "${extra[@]}"
