#ifndef GLOBAL_PARAMETERS_HPP
#define GLOBAL_PARAMETERS_HPP

namespace image_processing {

namespace parameters{

    /**
     * @brief parameters for supervoxels segmentation and intrisic parameter of the camera.
     * This parameters are default ones. Replace by desired values
     */
    struct supervoxel{
        static constexpr bool use_transform = false;
        static constexpr float voxel_resolution = 0.005f;
        static constexpr float color_importance = 0.2f;
        static constexpr float spatial_importance = 0.4f;
        static constexpr float normal_importance = 0.4f;
        static constexpr float seed_resolution = 0.05f;

        static constexpr float depth_princ_pt_x = 479.75;
        static constexpr float depth_princ_pt_y = 269.75;
        static constexpr float rgb_princ_pt_x = 479.75;
        static constexpr float rgb_princ_pt_y = 269.75;
        static constexpr float focal_length_x = 540.68603515625;
        static constexpr float focal_length_y = 540.68603515625;
        static constexpr float height = 540;
        static constexpr float width = 960;
    };

    /**
     * @brief The soi struct
     * @param increment value for a interesting surface
     * @param value for a non interesting surface
     * @param importance of the color and the normal in the distance, if it's value is 0.5 the both have the same importance.
     * @param threshold for considering if two surface are too different;
     */
    struct soi{
        static constexpr float interest_increment = 0.1f;
        static constexpr float non_interest_val = 0.f;
        static constexpr float color_normal_ratio = .5f;
//        static constexpr float normal_importance = .5f;
        static constexpr float distance_threshold = 0.5f;
    };
}

}

#endif //GLOBAL_PARAMETERS_HPP
