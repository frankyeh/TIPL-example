#include "TIPL/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{
    // 1: load example image
    tipl::image<3> hfrom;
    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }
    tipl::downsample_with_padding(hfrom);
    tipl::downsample_with_padding(hfrom);
    tipl::vector<3> voxel_size(1.0f,1.0f,1.0f);


    // 2: setup the transformation for linear registration algorithms to solve

    tipl::affine_transform<float> affine = {15.0,0,0,0.1f,0.02f,0,1.1f,1.2f,0.95f,0.02f,0,0};
    std::cout << "ground truth=\n" << affine;

    // 3: get the transformed image
    tipl::image<3> hto(hfrom.shape());
    tipl::transformation_matrix<float> trans0(tipl::affine_transform<float>(),
                                              hfrom.shape(),voxel_size,hto.shape(),voxel_size);
    tipl::transformation_matrix<float> trans(affine,hfrom.shape(),voxel_size,hto.shape(),voxel_size);
    tipl::resample_mt(hfrom,hto,trans);

    // 4: now use transformed image to calculate the transformation
    bool terminated = false;
    {
        std::cout << "\ncost function using cpu" << std::endl;
        tipl::reg::mutual_information mi;
        {
            tipl::time t("cpu time for cost function:");
            for(unsigned int i = 0;i < 20;++i)
                mi(hfrom,hto,trans0);
            std::cout << "cpu result:" << mi(hfrom,hto,trans0) << std::endl;
        }
        std::cout << "\ncost function using gpu" << std::endl;
        tipl::reg::mutual_information_cuda mi2;
        {
            tipl::time t("gpu time for cost function:");
            for(unsigned int i = 0;i < 20;++i)
                mi2(hfrom,hto,trans0);
            std::cout << "gpu result:" << mi2(hfrom,hto,trans0) << std::endl;
        }

    }

    {
        std::cout << "\nsolve using cpu" << std::endl;
        tipl::time t("cpu time (ms):");
        tipl::affine_transform<float> answer;
        tipl::reg::linear_two_way<tipl::reg::mutual_information> // use cpu multithread to calculate the cost function
                (hto,voxel_size,hfrom,voxel_size,answer,tipl::reg::affine,[&](void){return terminated;});
        std::cout << "cpu answer:\n" << answer;
    }

    {
        std::cout << "\nsolve using gpu" << std::endl;
        tipl::time t("gpu time (ms):");
        tipl::affine_transform<float> answer;
        tipl::reg::linear_two_way<tipl::reg::mutual_information_cuda> // use cuda to calculate the cost function
                (hto,voxel_size,hfrom,voxel_size,answer,tipl::reg::affine,[&](void){return terminated;});
        std::cout << "gpu answer:\n" << answer;
        std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
    }
    return 0;
}
