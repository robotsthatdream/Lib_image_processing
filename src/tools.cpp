#include <iostream>
#include <image_processing/tools.hpp>

//using namespace image_processing;


bool image_processing::tools::extract_convex_hull(pcl::PointCloud<PointT>::ConstPtr cloud,  std::vector<Eigen::Vector3d>& vertex_list){
    pcl::ConvexHull<PointT> hull_extractor;
    pcl::PointCloud<PointT> hull_cloud;
    hull_extractor.setInputCloud(cloud);
    hull_extractor.reconstruct(hull_cloud);

    if(hull_cloud.empty()){
        std::cerr << "unable to compute the convex hull" << std::endl;
        return false;
    }

    for(auto it = hull_cloud.points.begin(); it != hull_cloud.points.end(); ++it)
        vertex_list.push_back(Eigen::Vector3d(it->x,it->y,it->z));


    return true;
}

void image_processing::tools::rgb2hsv (int r, int g, int b, float& fh, float& fs, float& fv)
{
    // mostly copied from opencv-svn/modules/imgproc/src/color.cpp
    // revision is 4351
    const int hsv_shift = 12;

    static const int div_table[] =
    {
        0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211,
        130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632,
        65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412,
        43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693,
        32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782,
        26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223,
        21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991,
        18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579,
        16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711,
        14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221,
        13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006,
        11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995,
        10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141,
        10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410,
        9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777,
        8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224,
        8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737,
        7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304,
        7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917,
        6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569,
        6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254,
        6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968,
        5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708,
        5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468,
        5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249,
        5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046,
        5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858,
        4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684,
        4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522,
        4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370,
        4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229,
        4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096
    };
    int hr = 180, hscale = 15;
    int h, s, v = b;
    int vmin = b, diff;
    int vr, vg;

    v = std::max<int> (v, g);
    v = std::max<int> (v, r);
    vmin = std::min<int> (vmin, g);
    vmin = std::min<int> (vmin, r);

    diff = v - vmin;
    vr = v == r ? -1 : 0;
    vg = v == g ? -1 : 0;

    s = diff * div_table[v] >> hsv_shift;
    h = (vr & (g - b)) +
            (~vr & ((vg & (b - r + 2 * diff))
                    + ((~vg) & (r - g + 4 * diff))));
    h = (h * div_table[diff] * hscale +
         (1 << (hsv_shift + 6))) >> (7 + hsv_shift);

    h += h < 0 ? hr : 0;
    fh = static_cast<float> (h) / 180.0f;
    fs = static_cast<float> (s) / 255.0f;
    fv = static_cast<float> (v) / 255.0f;
}

void image_processing::tools::cloudRGB2HSV(const PointCloudT::Ptr input, PointCloudHSV::Ptr output){
    float h, s, v;

    for(auto itr = input->begin(); itr != input->end(); itr++){

        rgb2hsv(itr->r,itr->g,itr->b,h,s,v);
        output->push_back(PointHSV(h,s,v));
        output->back().x = itr->x;
        output->back().y = itr->y;
        output->back().z = itr->z;
    }

}

void image_processing::tools::rgb2Lab(int r, int g, int b, float& L, float& a, float& b2){

    float sRGB_LUT[256];
    float sXYZ_LUT[4000];

    for (int i = 0; i < 256; i++)
    {
        float f = static_cast<float> (i) / 255.0f;
        if (f > 0.04045)
            sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
        else
            sRGB_LUT[i] = f / 12.92f;
    }

    for (int i = 0; i < 4000; i++)
    {
        float f = static_cast<float> (i) / 4000.0f;
        if (f > 0.008856)
            sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
        else
            sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
    }


    float fr = sRGB_LUT[r];
    float fg = sRGB_LUT[g];
    float fb = sRGB_LUT[b];

    // Use white = D65
    const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
    const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
    const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

    float vx = x / 0.95047f;
    float vy = y;
    float vz = z / 1.08883f;

    vx = sXYZ_LUT[int(vx*4000)];
    vy = sXYZ_LUT[int(vy*4000)];
    vz = sXYZ_LUT[int(vz*4000)];

    L = 116.0f * vy - 16.0f;
    if (L > 100)
        L = 100.0f;

    L = L/100.0f;

    a = 500.0f * (vx - vy);
    if (a > 120)
        a = 120.0f;
    else if (a <- 120)
        a = -120.0f;

    a = a/120.0f;

    b2 = 200.0f * (vy - vz);
    if (b2 > 120)
        b2 = 120.0f;
    else if (b2<- 120)
        b2 = -120.0f;

    b2 = b2/120.0f;
}
