__kernel void resize_rgb_bilinear(
    __read_only image2d_t input,  // 输入图像 (CL_RGBA, CL_UNORM_INT8)
    __write_only image2d_t output, // 输出图像 (CL_RGBA, CL_UNORM_INT8)
    float scale_x,                // X 方向缩放比例
    float scale_y)                // Y 方向缩放比例
{
    // 获取输出图像的坐标
    int x = get_global_id(0);
    int y = get_global_id(1);

    // 计算输入图像中的对应坐标
    float in_x = (float)x * scale_x;
    float in_y = (float)y * scale_y;

    // 使用内置的采样器进行双线性插值
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // 使用非归一化坐标
                        CLK_ADDRESS_CLAMP_TO_EDGE |   // 边缘处理方式
                        CLK_FILTER_LINEAR;            // 双线性插值

    // 从输入图像中读取插值后的像素值
    float4 pixel = read_imagef(input, sampler, (float2)(in_x, in_y));

    // 将结果写入输出图像
    write_imagef(output, (int2)(x, y), pixel);
}