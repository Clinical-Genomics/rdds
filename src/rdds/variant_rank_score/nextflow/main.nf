params.version = "0.0.0-rc0"

println "running RDDS VRS version ${params.version}"

include {VRS_INFER} from './modules/rdds_vrs/infer/main.nf'

workflow{
    VRS_INFER()
}