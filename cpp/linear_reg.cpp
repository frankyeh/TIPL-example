#include "TIPL/tipl.hpp"
int main(void)
{
    tipl::image<3> Is,It; // image volume
    tipl::vector<3> vs_s,vs_t; // voxel size

    tipl::io::nifti nii_s,nii_t;

    std::cout << "loading template t1w" << std::endl;
    if(!nii_t.load_from_file("./TIPL-example/data/mni_icbm152_t1.nii"))
    {
        std::cout << "cannot open template t1w" << std::endl;
        return 1;
    }

    std::cout << "loading subject t1w" << std::endl;
    if(!nii_s.load_from_file("./TIPL-example/data/100206_T1w.nii"))
    {
        std::cout << "cannot open subject t1w" << std::endl;
        return 1;
    }

    nii_t >> It;
    nii_s >> Is;
    std::cout << nii_t << std::endl;

    nii_t.get_voxel_size(vs_t);
    nii_s.get_voxel_size(vs_s);

    tipl::affine_transform<double> T;
    std::cout << "running linear registration using mutual information" << std::endl;
    bool terminated = false;
    tipl::reg::linear(It,vs_t,Is,vs_s,T,
                                 tipl::reg::affine,
                                 tipl::reg::correlation(),
                                 terminated,0.01);

    std::cout << T;

    return 0;
}
