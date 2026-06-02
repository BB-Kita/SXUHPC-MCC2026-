#include <hdf5.h>
#include <iostream>
int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        const char* path = argv[a];
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (f < 0) { std::cerr << path << " open failed\n"; continue; }
        hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
        if (d < 0) { std::cerr << path << " data open failed\n"; H5Fclose(f); continue; }
        hid_t dcpl = H5Dget_create_plist(d);
        H5D_layout_t layout = H5Pget_layout(dcpl);
        haddr_t off = H5Dget_offset(d);
        hid_t space = H5Dget_space(d);
        int nd = H5Sget_simple_extent_ndims(space);
        hsize_t dims[4] = {0,0,0,0};
        H5Sget_simple_extent_dims(space, dims, nullptr);
        hid_t type = H5Dget_type(d);
        size_t type_size = H5Tget_size(type);
        std::cout << path << " layout=" << (int)layout << " offset=" << (unsigned long long)off
                  << " ndims=" << nd << " dims=" << dims[0] << "x" << dims[1]
                  << " type_size=" << type_size << "\n";
        H5Pclose(dcpl); H5Tclose(type); H5Sclose(space); H5Dclose(d); H5Fclose(f);
    }
}
