__kernel void nv21_split_and_assemble(__global uchar* y_plane, __global uchar* uv_plane, __global uchar* output_y, __global uchar* output_uv, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int half_width = width / 2;
    int half_height = height / 2;

    // Process Y plane
    if (y < height) {
        if (x < half_width) {
            // Top half
            output_y[y * width + x] = y_plane[y * width + x];
            output_y[(y + half_height) * width + x] = y_plane[y * width + x + half_width];
        }
    }

    // Process UV plane
    if (y < half_height) {
        if (x < half_width) {
            // Top half
            output_uv[y * width + x] = uv_plane[y * width + x];
            output_uv[(y + half_height) * width + x] = uv_plane[y * width + x + half_width];
        }
    }
}