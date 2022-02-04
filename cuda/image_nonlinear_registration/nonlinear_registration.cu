#include "TIPL/tipl.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T,typename U>
void check_dif(T& lhs,U& rhs)
{
    auto d = lhs;
    d -= tipl::host_image<3,tipl::vector<3> >(rhs);
    std::cout << "  sum of difference=" << tipl::sum(d) << std::endl;
    rhs = lhs;
}

int main(void)
{
    tipl::image<3> hfrom,hto;
    tipl::image<3,tipl::vector<3> > dis;

    // 1: load example image

    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }

    tipl::vector<3> voxel_size(1.0f,1.0f,1.0f);

    // 2: setup the transformation to generate a warpped image
    tipl::affine_transform<float> affine = {0,0,0,0,0,0,1.1f,1.2f,0.95f,0.01f,0,0.005f};
    std::cout << affine << std::endl;


    // 3: get the transformed image
    hto.resize(hfrom.shape());
    tipl::transformation_matrix<float> trans(affine,hfrom.shape(),tipl::v(1,1,1),hto.shape(),tipl::v(1,1,1));
    tipl::resample_mt(hfrom,hto,trans);

    std::cout << "image correlation:" << tipl::correlation(hfrom.begin(),hfrom.end(),hto.begin()) << std::endl;


    // 4: Copy host image to device
    tipl::device_image<3> dfrom(hfrom), dto(hto);
    tipl::device_image<3,tipl::vector<3> > ddis(hto.shape());


    std::cout << "\n==TEST 1: gradient calculation==" << std::endl;

    {
        dis.resize(hto.shape());
        tipl::time t("  cpu time on calculating the gradient (ms):");
        std::cout << "  r2=" << tipl::reg::cdm_get_gradient(hfrom,hto,dis) << std::endl;
    }
    {
        tipl::time t("  gpu time on calculating the gradient (ms):");
        std::cout << "  r2=" << tipl::reg::cdm_get_gradient_cuda(dfrom,dto,ddis) << std::endl;
    }
    check_dif(dis,ddis);

    std::cout << "\n==TEST 2: poisson solver==" << std::endl;

    {
        tipl::time t("  solve poisson using cpu:");
        bool terminated = false;
        tipl::reg::cdm_solve_poisson(dis,terminated);
    }
    {
        tipl::time t("  solve poisson using gpu:");
        bool terminated = false;
        tipl::reg::cdm_solve_poisson_cuda(ddis,terminated);
        cudaDeviceSynchronize();
    }
    check_dif(dis,ddis);


    std::cout << "\n==TEST 3: accumulate dis==" << std::endl;

    {
        tipl::time t("  accumulate displacement using cpu:");
        float theta = 0.0;
        bool terminated = false;
        tipl::reg::cdm_accumulate_dis(dis,dis,theta,0.5f);
        std::cout << "  theta=" << theta << std::endl;
    }

    {
        tipl::time t("  accumulate displacement using gpu:");
        float theta = 0.0;
        bool terminated = false;
        tipl::reg::cdm_accumulate_dis_cuda(ddis,ddis,theta,0.5f);
        std::cout << "  theta=" << theta << std::endl;
    }
    check_dif(dis,ddis);

    std::cout << "\n==TEST 4: displacement constraint==" << std::endl;

    {
        tipl::time t("  constraint displacement using cpu:");
        tipl::reg::cdm_constraint(dis,1.0f);
    }

    {
        tipl::time t("  constraint displacement using gpu:");
        tipl::reg::cdm_constraint_cuda(ddis,1.0f);
    }

    check_dif(dis,ddis);

    std::cout << "\n==Emsumble TEST: nonlinear registration==" << std::endl;

    {
        tipl::time t("  nonlinear registration using cpu:");
        bool terminated = false;
        dis.swap(decltype(dis)(dis.shape()));
        tipl::reg::cdm(hfrom,hto,dis,terminated);
    }

    {
        tipl::image<3> to_;
        tipl::compose_displacement(hto,dis,to_);
        std::cout << "  image correlation:" <<
            tipl::correlation(hfrom.begin(),hfrom.end(),to_.begin()) << std::endl;
        to_.save_to_file<tipl::io::nifti>("cpu_result.nii");
    }

    {
        tipl::time t("  nonlinear registration using gpu:");
        bool terminated = false;
        ddis.swap(decltype(ddis)(ddis.shape()));
        tipl::reg::cdm_cuda(dfrom,dto,ddis,terminated);

    }

    {
        tipl::image<3> to_;
        tipl::compose_displacement(hto,tipl::host_image<3,tipl::vector<3>>(ddis),to_);
        std::cout << "  image correlation:" <<
            tipl::correlation(hfrom.begin(),hfrom.end(),to_.begin()) << std::endl;
        to_.save_to_file<tipl::io::nifti>("gpu_result.nii");
    }

    return 0;
}
