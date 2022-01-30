#include "tipl/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{
    // 1: load atlas (integers representing brain regions)
    tipl::image<3> hfrom;
    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }
    // enlarge it
    tipl::upsampling(hfrom);
    tipl::upsampling(hfrom);

    tipl::image<3> hto(hfrom.shape());
    tipl::device_image<3> dto(hfrom.shape()),dfrom(hfrom);


    // use single thread
    {
        tipl::time t;

        for(size_t i = 0;i < hfrom.size();++i)
            if(hfrom[i] > 0)
                hto[i] = hfrom[i]*5.5f+100.0f;

        std::cout << "single thread time:" << t.elapsed<std::chrono::milliseconds>() << std::endl;
    }
    // use multi thread
    {
        tipl::time t;

        (hto = hfrom[hfrom > 0]*5.5f+100.0f)
                >> tipl::backend::mt();

        std::cout << "plain multithread time:" << t.elapsed<std::chrono::milliseconds>() << std::endl;
    }
    // use multi thread
    {
        tipl::time t;

        tipl::par_for(hfrom.size(),[&](size_t i)
        {
            if(hfrom[i] > 0)
                hto[i] = hfrom[i]*5.5f+100.0f;
        });
        std::cout << "par_for multi-thread time:" << t.elapsed<std::chrono::milliseconds>() << std::endl;
    }

    {
        auto from = tipl::make_alias(dfrom);
        auto to = tipl::make_alias(dto);
        tipl::time t;

        tipl::cuda_for(from.size(),[=]__device__(size_t i)
        {
           if(from[i] > 0)
               to[i] = from[i]*5.5f+100.0f;
        });

        cudaDeviceSynchronize();
        std::cout << "cuda_for time:" << t.elapsed<std::chrono::milliseconds>() << std::endl;
        std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
    }

    /*
     * {
        tipl::device_image<3> dto(hfrom);
        auto dto_alias = tipl::make_alias(dto);
        tipl::time t;
        // smooth each region
        std::cout << "gpu time:" << t.elapsed<std::chrono::milliseconds>() << std::endl;
        hto.save_to_file<tipl::io::nifti>("Brainnectome_atlas_smoothed.nii");
    }*/
    return 0;
}
