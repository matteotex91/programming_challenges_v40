#include <cmath>
#include <cstddef>
#include <vector>

// compile command :
// g++ -shared -o voronoi.so -fPIC voronoi.cpp
// this method calculates the voronoi map given a certain number of vectors in the RGB space

extern "C" void voronoi(const int *R, const int *G, const int *B, size_t point_count, size_t color_resolution, int *wrapped_voronoi_map)
{
    int diff_r = 0;
    int diff_g = 0;
    int diff_b = 0;
    int closest_color_index = 0;
    int closest_color_norm = -1;
    int current_norm = 0;
    for (size_t ix = 0; ix < color_resolution; ++ix)
    {
        for (size_t iy = 0; iy < color_resolution; ++iy)
        {
            for (size_t iz = 0; iz < color_resolution; ++iz)
            {
                closest_color_norm = -1;
                for (size_t color_index = 0; color_index < point_count; ++color_index)
                {
                    diff_r = R[color_index] - ix;
                    diff_g = G[color_index] - iy;
                    diff_b = B[color_index] - iz;
                    current_norm = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
                    if (closest_color_norm == -1)
                    {
                        closest_color_norm = current_norm;
                        closest_color_index = 0;
                    }
                    if (closest_color_norm > current_norm)
                    {
                        closest_color_norm = current_norm;
                        closest_color_index = color_index;
                    }
                }
                wrapped_voronoi_map[ix + color_resolution * (iy + color_resolution * iz)] = closest_color_index;
            }
        }
    }
}