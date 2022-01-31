#include "TIPL/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{

    // 1: load example image
    tipl::host_image<3> hfrom,hto;
    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }

    // 2: copy host image to device image
    tipl::device_image<3> dfrom(hfrom),dto;  //device memory image


    // 3: setup the transformation matrix
    tipl::transformation_matrix<float> trans;
    //scale by 0.8
    trans.sr[0] = 0.8;
    trans.sr[4] = 0.8;
    trans.sr[8] = 0.8;
    //shift x by 10 y by 20
    trans.shift[0] = 10;
    trans.shift[1] = 20;
    trans.shift[2] = 0;

    // 4: image resample using cpu
    hto.resize(hfrom.shape());
    {
        tipl::time t("cpu resample:");
        for(int i = 0;i < 100;++i)
            tipl::resample_mt(hfrom,hto,trans);
    }
    // save to file
    hto.save_to_file<tipl::io::nifti>("cpu.nii");

    // 5: image resample using gpu
    dto.resize(dfrom.shape());
    {
        tipl::time t("gpu resample time:");
        for(int i = 0;i < 100;++i)
            tipl::resample_cuda(dfrom,dto,trans);
    }
    // copy device image to host and save
    tipl::host_image<3>(dto).save_to_file<tipl::io::nifti>("gpu.nii");

    return 0;
}

