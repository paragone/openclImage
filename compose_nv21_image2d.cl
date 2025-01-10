__kernel void compose_nv21(
    __read_only image2d_t input_y1,
    __read_only image2d_t input_uv1,
    __read_only image2d_t input_y2,
    __read_only image2d_t input_uv2,
    __write_only image2d_t output_y,
    __write_only image2d_t output_uv,
    int width1,
    int width2,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int output_width = width1 + width2;

    if (y >= height) return;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE |
                        CLK_FILTER_LINEAR;

    // 读取 Y 通道
    uint y_pixel;
    if (x < width1) {
        y_pixel = read_imageui(input_y1, sampler, (int2)(x, y)).x;
    } else {
        y_pixel = read_imageui(input_y2, sampler, (int2)(x - width1, y)).x;
    }

    // 写入 Y 通道
    write_imageui(output_y, (int2)(x, y), (uint4)(y_pixel, 0, 0, 0));

    // 处理 UV 通道
    if (x % 2 == 0 && y % 2 == 0) {
        int uv_x = x / 2;
        int uv_y = y / 2;

        uint2 uv_pixel;
        if (uv_x < width1 / 2) {
            uv_pixel = read_imageui(input_uv1, sampler, (int2)(uv_x, uv_y)).xy;
        } else {
            uv_pixel = read_imageui(input_uv2, sampler, (int2)(uv_x - width1 / 2, uv_y)).xy;
        }

        // 写入 UV 通道
        write_imageui(output_uv, (int2)(uv_x, uv_y), (uint4)(uv_pixel.x, uv_pixel.y, 0, 0));
    }
}