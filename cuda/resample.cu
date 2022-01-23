#include "tipl/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{
    tipl::image<3,float,tipl::host_memory> hfrom,hto;			 // host memory image
    hfrom.load_from_file<tipl::io::nifti>("example_t1w.nii");
    hto.resize(hfrom.shape());

    tipl::image<3,float,tipl::device_memory> dfrom,dto;  //device memory image
    dfrom = hfrom;
    dto.resize(dfrom.shape());

    tipl::transformation_matrix<float> trans;
    trans.sr[0] = 0.8;  //scale by 0.8
    trans.sr[4] = 0.8;
    trans.sr[8] = 0.8;
    trans.shift[9] = 10;  //shift x by 10
    trans.shift[10] = 20; // shift y by 20
    trans.shift[11] = 0;

    std::cout << "" << std::endl;
    auto p = std::chrono::high_resolution_clock::now();
    {
       tipl::resample_cuda(dfrom,dto,trans);
    }
    
    std::cout << "gpu resample time:" << std::chrono::duration_cast<std::chrono::microseconds>
                 (std::chrono::high_resolution_clock::now()-p).count() << std::endl;


    p = std::chrono::high_resolution_clock::now();
    {
       tipl::resample_mt(hfrom,hto,trans);
    }
    std::cout << "cpu resample:" << std::chrono::duration_cast<std::chrono::microseconds>
                 (std::chrono::high_resolution_clock::now()-p).count() << std::endl;
    

    hto.save_to_file<tipl::io::nifti>("cpu.nii");

    hto = dto;

    hto.save_to_file<tipl::io::nifti>("gpu.nii");
}

