#include "TIPL/tipl.hpp"
int main(void)
{
    tipl::image<3> I;
    tipl::io::nifti nii;
    if(!nii.load_from_file("./TIPL-example/data/mni_icbm152_t1.nii"))
    {
        std::cout << "cannot open file" << std::endl;
        return 1;  
    }    
    nii >> I;
    std::cout << nii << std::endl;
    return 0;
}