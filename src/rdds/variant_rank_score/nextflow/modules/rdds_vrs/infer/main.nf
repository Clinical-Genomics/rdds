
process VRS_INFER {
    debug false

    def version = 'vrs-testnextflow'
    println "Running VRS version: ${version}"

    stageInMode('copy')
    publishDir "${params.output_dir}", mode: 'copy', overwrite: true, failOnError: true

    input:
    path input_vcf
    path output_dir

    output:
    path '*-predictions.vcf', emit: vcf_with_inferences

    container "docker.io/clinicalgenomics/rdds_vrs:${version}"

    script:
    """
    export PYTHONPATH=/rdds/src
    . /opt/pyenv/bin/activate
    python3 -m rdds.variant_rank_score predict-on-vcf --cpu_cores ${task.cpus} ${input_vcf}
    """
}