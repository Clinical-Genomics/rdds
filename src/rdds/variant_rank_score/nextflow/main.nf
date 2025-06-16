params.output_dir = "$launchDir/tmp/vrs"

include {VRS_INFER} from './modules/rdds_vrs/infer/main.nf'

process create_output_dir {
    input:
    path dir

    script:
    println "Output directory: $dir"
    """
    mkdir ${dir}
    """
}

workflow{
    //create_output_dir(params.output_dir)

    // Run inference on input VCFs (supports globbing)
    def input_vcfs = channel.fromPath('../../../tests/variant_rank_score/test_data.vcf')
    VRS_INFER(input_vcfs, params.output_dir)
}