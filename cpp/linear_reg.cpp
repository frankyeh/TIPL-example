#include "TIPL/tipl.hpp"
int main(void)
{
    tipl::image<3> Is,It; // image volume
    tipl::vector<3> vs_s,vs_t; // voxel size

    tipl::io::nifti nii_s,nii_t;
    if(!nii_t.load_from_file("./TIPL-example/data/mni_icbm152_t1.nii") ||
       !nii_s.load_from_file("./TIPL-example/data/100206_T1w.nii"))
    {
        std::cout << "cannot open data" << std::endl;
        return 1;
    }

    nii_t >> It;
    nii_s >> Is;
    std::cout << nii_t << std::endl;

    nii_t.get_voxel_size(vs_t);
    nii_s.get_voxel_size(vs_s);

    tipl::affine_transform<float> T;
    std::cout << "running linear registration using correlation" << std::endl;
    bool terminated = false;
    tipl::reg::linear<tipl::reg::correlation>(It,vs_t,Is,vs_s,T,
                                 tipl::reg::affine,
                                 terminated,0.01);

    std::cout << T;

    return 0;
}
