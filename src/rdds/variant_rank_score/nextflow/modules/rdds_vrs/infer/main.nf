process VRS_INFER {
    debug true
    def version = 'v1.11.0-rc3'
    println "Running VRS version: ${version}"

    // TODO: Adjust to 10cores - 125GB RAM
    cpus 5
    memory '75 GB'
    publishDir "${params.output_dir}", mode: 'copy', overwrite: true, failOnError: true
    //container 'docker.io/clinicalgenomics/rdds_vrs:v1.11.0-rc3'

    input:
    path input_vcf
    path output_dir

    output:
    path '*.vcf', emit: vcf_with_inferences


    script:
    """
    docker run \
    -t \
    --rm \
    -v ${workDir}:/data docker.io/clinicalgenomics/rdds_vrs:$version \
    --cpu_cores ${task.cpus} \
    /data/test_data.vcf
    """
}