// Example workflow to show usage of vrs nextflow module

params.output_dir = "$launchDir/tmp/vrs"

include {VRS_INFER} from './modules/rdds_vrs/infer/main.nf'

workflow{
    // Run inference on input VCFs (supports globbing)
    def input_vcfs = channel.fromPath('../../../tests/variant_rank_score/test_data.vcf')
    VRS_INFER(input_vcfs, params.output_dir)
}