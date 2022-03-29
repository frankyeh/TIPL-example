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
    tipl::image<3,tipl::vector<3> > dis,inv_dis;

    // 1: load example image

    if(!hfrom.load_from_file<tipl::io::nifti>("100206_T1w.nii") ||
       !hto.load_from_file<tipl::io::nifti>("mni_icbm152_t1.nii"))
    {
        std::cout << "cannot find the sample file" << std::endl;
        return 1;
    }
    tipl::filter::gaussian(hfrom);
    tipl::filter::gaussian(hfrom);
    tipl::filter::gaussian(hto);

    std::cout << "initial correlation:" << tipl::correlation(hfrom.begin(),hfrom.end(),hto.begin()) << std::endl;


    // 4: Copy host image to device
    tipl::device_image<3> dfrom(hfrom), dto(hto);
    tipl::device_image<3,tipl::vector<3> > ddis(hto.shape()),inv_ddis(hto.shape());


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
        tipl::time t("  maximum displacement using cpu:");
        float theta = tipl::reg::cdm_max_displacement_length(dis);
        std::cout << "  theta=" << theta << std::endl;
    }

    {
        tipl::time t("  maximum displacement using gpu:");
        float theta = tipl::reg::cdm_max_displacement_length_cuda(ddis);
        std::cout << "  theta=" << theta << std::endl;
    }
    check_dif(dis,ddis);

    std::cout << "\n==TEST 4: displacement constraint==" << std::endl;

    {
        tipl::time t("  constraint displacement using cpu:");
        tipl::reg::cdm_constraint(dis);
    }

    {
        tipl::time t("  constraint displacement using gpu:");
        tipl::reg::cdm_constraint_cuda(ddis);
    }

    check_dif(dis,ddis);


    std::cout << "\n==TEST 5: invert displacement==" << std::endl;
    ddis = dis;
    {
        inv_dis.clear();
        inv_dis.resize(dis.shape());
        tipl::time t("  inverse displacement using cpu:");
        tipl::invert_displacement_imp(dis,inv_dis);
    }

    {
        inv_ddis.clear();
        inv_ddis.resize(dis.shape());
        tipl::time t("  inverse displacement using gpu:");
        tipl::invert_displacement_cuda_imp(ddis,inv_ddis);
    }

    check_dif(inv_dis,inv_ddis);


    std::cout << "\n==Ensemble TEST: nonlinear registration==" << std::endl;

    {
        tipl::time t("  nonlinear registration using cpu:");
        bool terminated = false;

 				tipl::image<3,tipl::vector<3> > dis_tmp(dis.shape());
        dis.swap(dis_tmp);
        tipl::reg::cdm(hfrom,hto,dis,inv_dis,terminated);
    }

    {
        tipl::image<3> from_;
        tipl::compose_displacement(hfrom,inv_dis,from_);
        std::cout << "  image correlation:" <<
            tipl::correlation(hto.begin(),hto.end(),from_.begin()) << std::endl;
        from_.save_to_file<tipl::io::nifti>("cpu_result.nii");
    }

    {
        tipl::time t("  nonlinear registration using gpu:");
        bool terminated = false;
				tipl::device_image<3,tipl::vector<3> > ddis_tmp(ddis.shape());
        ddis.swap(ddis_tmp);
        tipl::reg::cdm_cuda(dfrom,dto,ddis,inv_ddis,terminated);

    }

    {
        tipl::image<3> from_;
        tipl::compose_displacement(hfrom,tipl::host_image<3,tipl::vector<3> >(inv_ddis),from_);
        std::cout << "  image correlation:" <<
            tipl::correlation(hto.begin(),hto.end(),from_.begin()) << std::endl;
        from_.save_to_file<tipl::io::nifti>("gpu_result.nii");
    }

    return 0;
}
