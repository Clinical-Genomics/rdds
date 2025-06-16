process VRS_INFER {
    def version = 'v1.11.0-rc3'
    println "Running VRS version: ${version}"

    publishDir "${params.output_dir}", mode: 'copy', overwrite: true
    //container 'docker.io/clinicalgenomics/rdds_vrs:v1.11.0-rc3'

    input:
    path input_vcf
    path output_dir

    output:
    path '*.vcf', emit: vcf_with_inferences


    script:
    """
    touch asd.vcf
    """
}