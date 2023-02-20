#include "TIPL/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main(void)
{
    // read a NIFTI file
    tipl::image<3> hfrom;
    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample image 100206_T1w.nii" << std::endl;
        return 1;
    }
    // enlarge it
    tipl::upsampling(hfrom);
    tipl::upsampling(hfrom);

    tipl::image<3> hto(hfrom.shape());

    // use single thread
    {
        tipl::time t("single thread time:");
        for(size_t i = 0;i < hfrom.size();++i)
            if(hfrom[i] > 0)
                hto[i] = hfrom[i]*5.5f+100.0f;
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
        tipl::device_image<3> dto(hfrom.shape()),dfrom(hfrom.shape());
        {
            tipl::time t("cuda copy to gpu time:");
            dfrom = hfrom;
        }
        {
            tipl::time t("cuda computation time:");

            // create shared images that can be copied by the device lambda
            auto from = tipl::make_shared(dfrom);
            auto to = tipl::make_shared(dto);
            tipl::cuda_for(from.size(),[=]__device__(size_t i) mutable
            {
                if(from[i] > 0)
                   to[i] = from[i]*5.5f+100.0f;
            });
            // wait until the above code to be finished on the gpu.
            cudaDeviceSynchronize();
        }
        {
            tipl::time t("cuda copy from gpu time:");
            tipl::host_image<3> h(dto);
        }
        std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;

    }
    return 0;
}
