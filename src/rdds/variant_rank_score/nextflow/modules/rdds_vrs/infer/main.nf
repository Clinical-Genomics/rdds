process VRS_INFER {
    //tag "$meta.id"

    container 'docker.io/clinicalgenomics/rdds_vrs:v1.11.0-rc3'

    script:
    """
    echo hello
    """
}