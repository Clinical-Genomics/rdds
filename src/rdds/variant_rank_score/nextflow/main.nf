// Example workflow to show usage of vrs nextflow module

params.output_dir = "$launchDir/tmp/vrs"

include {VRS_INFER} from './modules/rdds_vrs/infer/main.nf'

process create_output_dir() {
    """
    mkdir -p ${params.output_dir}
    """
}

workflow{
    create_output_dir()
    // Run inference on input VCFs (supports globbing)
    def input_vcf = file('../../../tests/variant_rank_score/test_data.vcf')
    VRS_INFER(input_vcf, params.output_dir)
    VRS_INFER.out.vcf_with_inferences.view()
}