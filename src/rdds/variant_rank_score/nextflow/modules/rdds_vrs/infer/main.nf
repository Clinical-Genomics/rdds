// Nextflow does not support docker containers with set ENTRYPOINTs, https://nextflow.io/docs/latest/container.html

process VRS_INFER {
    debug true
    def version = 'vrs-testnextflow'
    println "Running VRS version: ${version}"

    stageInMode('copy')  // Copy files into workDir instead of symlinking

    input:
    path input_vcf
    path output_dir

    output:
    path '*-predictions.vcf', emit: vcf_with_inferences

    //println "Input VCF files: ${input_vcf}"

    // TODO: Adjust to 10cores - 125GB RAM
    cpus 5
    memory '75 GB'
    publishDir "${params.output_dir}", mode: 'copy', overwrite: true, failOnError: true
    container "docker.io/clinicalgenomics/rdds_vrs:${version}"
    //containerOptions "-v ${task.workDir}:/data"

    //def docker_mount_dir = input_vcf.getParent()

    script:
    """
    realpath .
    ls `dirname ${input_vcf}`
    stat ${input_vcf}

    export PYTHONPATH=/rdds/src
    . /opt/pyenv/bin/activate
    python3 -m rdds.variant_rank_score predict-on-vcf ${input_vcf}
    """
}
    //export PYTHONPATH=/rdds/src
    //python3 -m pytest /rdds/src/tests/variant_rank_score -k test_inference
    //python3 -m rdds.variant_rank_score predict-on-vcf \$@