#include "tipl/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{

    // 1: load example image
    tipl::image<3,float,tipl::host_memory> hfrom,hto;
    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }

    // 2: copy host image to device image
    tipl::image<3,float,tipl::device_memory> dfrom(hfrom),dto;  //device memory image


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
    tipl::time t;
    {
        hto.resize(hfrom.shape());
        tipl::resample_mt(hfrom,hto,trans);
    }
    std::cout << "cpu resample:" << t.elapsed<std::chrono::microseconds>() << std::endl;
    // save to file
    hto.save_to_file<tipl::io::nifti>("cpu.nii");

    // 5: image resample using gpu
    t.restart();
    {
        dto.resize(dfrom.shape());
        tipl::resample_cuda(dfrom,dto,trans);
    }
    std::cout << "gpu resample time:" << t.elapsed<std::chrono::microseconds>() << std::endl;
    // copy device image to host and save
    tipl::image<3,float,tipl::host_memory>(dto).save_to_file<tipl::io::nifti>("gpu.nii");

    return 0;
}

