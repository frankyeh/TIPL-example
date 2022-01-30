#include "TIPL/tipl.hpp"
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
        tipl::time t("single thread time:");

        for(size_t i = 0;i < hfrom.size();++i)
            if(hfrom[i] > 0)
                hto[i] = hfrom[i]*5.5f+100.0f;

    }
    // use multi thread
    {
        tipl::time t("simple multithread time:");

        (hto = hfrom[hfrom > 0]*5.5f+100.0f)
                >> tipl::backend::mt();

    }
    // use multi thread
    {
        tipl::time t("par_for multi-thread time:");

        tipl::par_for(hfrom.size(),[&](size_t i)
        {
            if(hfrom[i] > 0)
                hto[i] = hfrom[i]*5.5f+100.0f;
        });

    }
    // use cuda
    {
        auto from = tipl::make_shared(dfrom);
        auto to = tipl::make_shared(dto);
        tipl::time t("cuda_for time:");

        tipl::cuda_for(from.size(),[=]__device__(size_t i) mutable
        {
            if(from[i] > 0)
               to[i] = from[i]*5.5f+100.0f;
        });
        std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
    }

    return 0;
}
