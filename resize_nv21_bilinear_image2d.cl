__kernel void resize_nv21_bilinear(
    __read_only image2d_t input_y,  // Y 通道 (CL_R, CL_UNORM_INT8)
    __read_only image2d_t input_uv, // UV 通道 (CL_RG, CL_UNORM_INT8)
    __write_only image2d_t output_y, // 输出 Y 通道 (CL_R, CL_UNORM_INT8)
    __write_only image2d_t output_uv, // 输出 UV 通道 (CL_RG, CL_UNORM_INT8)
    float scale_x,                  // X 方向缩放比例
    float scale_y)                  // Y 方向缩放比例
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float in_x = (float)x * scale_x;
    float in_y = (float)y * scale_y;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE |
                        CLK_FILTER_LINEAR;

    // 读取 Y 通道
    uint y_pixel = read_imageui(input_y, sampler, (float2)(in_x, in_y)).x;

    // 读取 UV 通道
    uint2 uv_pixel = read_imageui(input_uv, sampler, (float2)(in_x / 2, in_y / 2)).xy;

    // 写入 Y 通道
    write_imageui(output_y, (int2)(x, y), (uint4)(y_pixel, 0, 0, 0));

    // 写入 UV 通道
    write_imageui(output_uv, (int2)(x / 2, y / 2), (uint4)(uv_pixel.x, uv_pixel.y, 0, 0));
}